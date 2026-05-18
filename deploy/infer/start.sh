#!/bin/bash

set -e  # Exit on error

ORIGINAL_ARGS=("$@")
for i in "${!ORIGINAL_ARGS[@]}"; do
    if [ "${ORIGINAL_ARGS[$i]}" = "--model" ]; then
        ORIGINAL_ARGS[$i]="--model-path"
    fi
done

# 环境变量说明:
# ENABLE_RDMA_GID_CHECK: 控制是否启用RDMA GID检查 (默认: 1=启用, 0=禁用)
# ENABLE_PD_GUARDIAN: 控制是否启用Guardian进程 (默认: 1=启用, 0=禁用)

# GID配置文件
LAST_GID_FILE="/tmp/last_v2_gid.txt"
CURRENT_GID_FILE="/tmp/current_v2_gid.txt"
#动态提取IB网卡
EXCLUDE_HCA=$(echo "${NCCL_IB_HCA:-mlx5_gdr_0}" | cut -d':' -f1 | cut -d',' -f1)
echo "EXCLUDE_HCA=$EXCLUDE_HCA"
echo "ENABLE_RDMA_GID_CHECK=${ENABLE_RDMA_GID_CHECK:-1}"

# Global variables
LOG_DIR="./logs/sglang"
LOG_FILE="${LOG_DIR}/sglang_monitor.log"
MAX_LOG_SIZE=$((100*1024*1024))
MIN_DISK_SPACE=$((500*1024))  # 500MB minimum
RANK_ID=${RANK_ID:-0}
LOG_RETENTION_DAYS=7
MAX_LOG_FILES=50

mkdir -p ${LOG_DIR}/moon

log_message() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "\033[32m[${timestamp}] $*\033[0m" | tee -a "${LOG_FILE}"
}

# 获取目标RDMA GID
get_rdma_target_gid() {
    show_gids | grep -w 'v2' | grep -w $EXCLUDE_HCA|tail -n 1 | awk '{print $3}'
}

# Add system info logging
log_system_info() {
    log_message "=== System Information ==="
    log_message "Hostname: $(hostname)"
    log_message "OS: $(uname -a)"
    log_message "Memory: $(free -h)"
    log_message "Disk Space: $(df -h /)"
    log_message "GPU Info: $(nvidia-smi --query-gpu=gpu_name,memory.total --format=csv,noheader 2>/dev/null || echo 'No GPU')"
    log_message "======================="
    log_message "Running on RANK_ID: ${RANK_ID}"

}

log_message "=== Command Arguments ==="
log_message "Number of arguments: $#"
log_message "All arguments: $@"
log_message "======================="

# Cleanup function
cleanup() {
    log_message "Received shutdown signal, cleaning up..."
    kill_sglang
    exit 0
}

# Signal handling
trap cleanup SIGTERM SIGINT

# Improved log rotation
rotate_log() {
    # Rotate current log if too large
    if [ -f "${LOG_FILE}" ]; then
        local size
        size=$(stat --format=%s "${LOG_FILE}" 2>/dev/null || stat -f%z "${LOG_FILE}")
        if [ "$size" -gt "${MAX_LOG_SIZE}" ]; then
            mv "${LOG_FILE}" "${LOG_FILE}.$(date +%Y%m%d-%H%M%S)"
        fi
    fi

    # Clean up old log files
    find "${LOG_DIR}" -name "sglang_monitor.log.*" -type f -mtime +${LOG_RETENTION_DAYS} -delete

    # If still too many files, delete oldest
    local log_count
    log_count=$(find "${LOG_DIR}" -name "sglang_monitor.log.*" | wc -l)
    if [ "$log_count" -gt "$MAX_LOG_FILES" ]; then
        find "${LOG_DIR}" -name "sglang_monitor.log.*" -type f -printf '%T+ %p\n' | \
        sort | head -n $(($log_count - $MAX_LOG_FILES)) | cut -d' ' -f2- | xargs rm -f
    fi

    log_message "Log rotation and cleanup completed"
}

# Check disk space
check_disk_space() {
    local available
    available=$(df -k "${LOG_DIR}" | awk 'NR==2 {print $4}')
    if [ "$available" -lt "$MIN_DISK_SPACE" ]; then
        log_message "WARNING: Low disk space"
        return 1
    fi
    log_message "check_disk_space done"

    return 0
}



# Improved process management
kill_sglang() {
    log_message "Attempting to kill sglang processes..."
    # 强杀python
    timeout 10 pkill -9 python3

    local pid
    pid=$(pgrep -f "/usr/local/bin/sglang" || echo "")

    if [ -n "$pid" ]; then
        timeout 10 kill -TERM $pid 2>/dev/null || true
        sleep 2
        if kill -0 $pid 2>/dev/null; then
            timeout 5 kill -9 $pid 2>/dev/null || true
        fi
    fi


    if pgrep -f "/usr/local/bin/sglang" >/dev/null; then
        log_message "ERROR: Failed to kill all processes"
        return 1
    fi
    return 0
}



# 检测变化 (返回: 0=变化 1=未变化 2=错误)
check_rdma_gid() {
    current_gid=$(get_rdma_target_gid)
    [ -z "$current_gid" ] && return 2

    echo "$current_gid" > "$CURRENT_GID_FILE"

    if [ ! -f "$LAST_GID_FILE" ]; then
        cp "$CURRENT_GID_FILE" "$LAST_GID_FILE"
        return 1
    fi

    if ! diff -q "$LAST_GID_FILE" "$CURRENT_GID_FILE" >/dev/null; then
        mv "$CURRENT_GID_FILE" "$LAST_GID_FILE"
        return 0
    fi

    return 1
}



# Improved service start
start_service() {
    log_message "Rotating logs if needed..."
    rotate_log
    # 启动时 mooncake 自适应机器GID
    if [ "${ENABLE_RDMA_GID_CHECK:-1}" = "1" ]; then
        export MC_GID_INDEX=$(get_rdma_target_gid)
        log_message "MC_GID_INDEX set to: $MC_GID_INDEX"
    else
        log_message "RDMA GID check is disabled, skipping MC_GID_INDEX setup"
    fi
    log_message "Starting sglang service..."
    if ! check_disk_space; then
        log_message "ERROR: Insufficient disk space"
        return 1
    fi

    {
        cmd=(sglang serve "$@")
        log_message "Executing command: $(printf '%q ' "${cmd[@]}")"
        ("${cmd[@]}" 2>&1 | while IFS= read -r line; do
            log_message "[SERVICE] $line"
        done) &
        pid=$!
        wait $pid && status=$? || status=$?
        log_message "Service exited with status: $status"
        return $status
    }

    PID=$!
    log_message "Service started with PID: $PID"

    # Add trap for cleanup
    trap 'log_message "Stopping service (PID: $PID)"; kill $PID 2>/dev/null' EXIT

    # Monitor startup
    sleep 5
    if ! ps -p $PID > /dev/null; then
        log_message "ERROR: Service failed to start"
        return 1
    fi

    log_message "Service startup complete"
    sleep 150  # Wait for GPU memory allocation
    return 0
}

check_gpu_usage() {
    log_message "Checking GPU memory usage on RANK ${RANK_ID}..."

    local total_mem=0

    # Read GPU memory values into array
    readarray -t gpu_mems < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo "0")

    # Sum up memory from all GPUs
    for mem in "${gpu_mems[@]}"; do
        total_mem=$((total_mem + mem))
    done

    log_message "Total GPU memory in use: ${total_mem}MB"

    # Check if total memory > 1GB
    if [ "$total_mem" -gt 2024 ]; then
        return 0
    else
        return 1
    fi
}

# Start Guardian process
start_guardian() {
    log_message "Starting Guardian process..."

    # Check if Guardian script exists
    local guardian_script="/sgl-workspace/guard/guarding.py"
    if [ ! -f "$guardian_script" ]; then
        log_message "WARNING: Guardian script not found at $guardian_script"
        return 1
    fi

    # Set default CLUSTER_IPS if not provided
    if [ -z "$CLUSTER_IPS" ]; then
        export CLUSTER_IPS="127.0.0.1"
        log_message "INFO: CLUSTER_IPS not set, using default: $CLUSTER_IPS"
    fi

    # Start Guardian in background
    {
        python3 "$guardian_script" 2>&1 | while IFS= read -r line; do
            log_message "[GUARDIAN] $line"
        done
    } &

    GUARDIAN_PID=$!
    log_message "Guardian started with PID: $GUARDIAN_PID"

    # Add Guardian to cleanup trap
    trap 'log_message "Stopping Guardian (PID: $GUARDIAN_PID)"; kill $GUARDIAN_PID 2>/dev/null; log_message "Stopping service (PID: $PID)"; kill $PID 2>/dev/null' EXIT

    # Wait a moment to check if Guardian started successfully
    sleep 2
    if ! ps -p $GUARDIAN_PID > /dev/null; then
        log_message "ERROR: Guardian failed to start"
        return 1
    fi

    log_message "Guardian startup complete"
    return 0
}

# Main loop
log_message "Starting deployment script"
log_system_info
start_service "${ORIGINAL_ARGS[@]}"

# Check if Guardian should be started based on environment variable
if [ "${ENABLE_PD_GUARDIAN:-1}" = "1" ]; then
    log_message "Guardian is enabled, starting Guardian process..."
    if ! start_guardian; then
        log_message "WARNING: Guardian failed to start, continuing without cluster protection"
    fi
else
    log_message "Guardian is disabled by ENABLE_PD_GUARDIAN environment variable"
fi

restart_service() {
    log_message "Restarting service..."
    kill_sglang
    sleep 20
    if ! start_service "${ORIGINAL_ARGS[@]}"; then
        log_message "ERROR: Failed to restart service"
        sleep 30
    fi
}

while true; do

    rotate_log

    # GID 重启逻辑
    if [ "${ENABLE_RDMA_GID_CHECK:-1}" = "1" ]; then
        if check_rdma_gid; then
            log_message "GID变化: $(cat "$LAST_GID_FILE")"
            restart_service
        elif [ $? -eq 1 ]; then
            log_message "GID未变化: $(cat "$LAST_GID_FILE")"
        else
            log_message "Warning: 获取GID失败"
        fi
    else
        log_message "RDMA GID check is disabled by ENABLE_RDMA_GID_CHECK environment variable"
    fi

    # 崩溃重启逻辑
    if ! check_gpu_usage; then
        log_message "GPU memory usage too low, restarting service..."
        restart_service
    fi

    sleep 8
done >> "${LOG_FILE}" 2>&1
