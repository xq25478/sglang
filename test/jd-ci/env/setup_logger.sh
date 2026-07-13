#!/bin/bash
# JD CI 统一日志工具。用法:
#   source env/setup_logger.sh <MODULE_NAME>
#   然后使用 jd_log 系列函数

MODULE_NAME="${1:-unknown}"
JD_LOG_DIR="${JD_LOG_DIR:-${SCRIPT_DIR}/../ci_logs}"
JD_LOG_FILE="${JD_LOG_DIR}/${MODULE_NAME}_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "${JD_LOG_DIR}"

# 同时输出到 stdout 和日志文件
exec 3>&1  # 保存原始 stdout

jd_log() {
    local level="${1:-INFO}"
    shift
    local ts=$(date '+%Y-%m-%d %H:%M:%S')
    local msg="[${ts}] [${MODULE_NAME}] [${level}] $*"
    echo "${msg}" | tee -a "${JD_LOG_FILE}"
}

jd_log_info()  { jd_log INFO "$@"; }
jd_log_warn()  { jd_log WARN "$@"; }
jd_log_error() { jd_log ERROR "$@"; }
jd_log_step()  { jd_log "====" "$@"; }

jd_log_separator() {
    local char="${1:-=}"
    local width="${2:-60}"
    local sep=$(printf "%${width}s" | tr ' ' "${char}")
    echo "${sep}" | tee -a "${JD_LOG_FILE}"
}

jd_log_header() {
    jd_log_separator "="
    jd_log "====" "JD CI Pipeline: ${MODULE_NAME}"
    jd_log "====" "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    jd_log_separator "="
}

jd_log_footer() {
    local exit_code="${1:-$?}"
    jd_log_separator "="
    jd_log "====" "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
    jd_log "====" "退出码: ${exit_code}"
    jd_log_separator "="
}

jd_log_file() {
    echo "${JD_LOG_FILE}"
}

# 执行命令并捕获输出到日志
jd_log_exec() {
    local desc="${1:-command}"
    jd_log_info "执行: ${desc}"
    shift
    "$@" 2>&1 | tee -a "${JD_LOG_FILE}"
    local ret=${PIPESTATUS[0]}
    if [ ${ret} -ne 0 ]; then
        jd_log_error "${desc} 失败 (exit=${ret})"
    else
        jd_log_info "${desc} 完成"
    fi
    return ${ret}
}

export JD_LOG_FILE
