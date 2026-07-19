#!/bin/bash
set -euo pipefail

print_usage() {
    cat <<'EOF'
用法:
  test/jd-ci/run_jd_ci.sh [-r | -m | -t | -h]

执行模式（互斥）:
  -r, --review       正常 PR 评审；默认模式。
                     强制编译 SGL-Kernel 和 Mooncake，更新正式缓存，
                     固定执行全部 JD CI 回归，不产出镜像。
                     兼容事件名: note__merge_request

  -m, --merge        快速复用正式缓存产出镜像。
                     任意分支都可以复用对应版本主分支的正式缓存；
                     缓存缺失立即失败，不执行 JD CI 回归，
                     产出带当前 commit 标识的 SGLang 镜像。
                     兼容事件名: merge_request__merged

  -t, --temp-image   临时分支验证镜像。
                     组件默认在 commit 独立临时目录中编译并安装；
                     可由用户显式跳过组件编译或全部测试；
                     所有已启用门禁通过后产出临时 SGLang 镜像并清理临时目录。

  -h, --help         显示本帮助并退出。

临时镜像选项（仅用于 -t，只接受 0 或 1）:
  JD_CI_SKIP_SGL_KERNEL_BUILD  默认 0；1 表示继承基础镜像中的 SGL-Kernel。
  JD_CI_SKIP_MOONCAKE_BUILD    默认 0；1 表示继承基础镜像中的 Mooncake。
  JD_CI_SKIP_TEST              默认 0；1 表示跳过全部 JD CI 回归。

约束:
  * 无参数等价于 -r。
  * -r 和 -m 不允许通过上述变量跳过正式流程。
  * -m 允许任意分支只读复用对应版本主分支的正式缓存；cache miss 立即失败。
  * -t 不能在正式分支运行，且不会读取、写入或覆盖正式组件缓存。
  * -t 的任一组件、容器或测试失败时，不推送临时 SGLang 镜像。

示例:
  test/jd-ci/run_jd_ci.sh -r
  test/jd-ci/run_jd_ci.sh note__merge_request
  test/jd-ci/run_jd_ci.sh -m
  test/jd-ci/run_jd_ci.sh merge_request__merged
  test/jd-ci/run_jd_ci.sh -t
  JD_CI_SKIP_SGL_KERNEL_BUILD=1 test/jd-ci/run_jd_ci.sh -t
  JD_CI_SKIP_MOONCAKE_BUILD=1 test/jd-ci/run_jd_ci.sh -t
  JD_CI_SKIP_SGL_KERNEL_BUILD=1 JD_CI_SKIP_MOONCAKE_BUILD=1 \
    test/jd-ci/run_jd_ci.sh -t
  JD_CI_SKIP_TEST=1 test/jd-ci/run_jd_ci.sh -t
EOF
}

usage_error() {
    echo "[SGLang CI] ERROR: $1" >&2
    print_usage >&2
    exit 2
}

validate_binary_option() {
    local name="$1"
    local value="$2"
    case "${value}" in
        0|1)
            ;;
        *)
            usage_error "${name} 只接受 0 或 1，当前值为 ${value}"
            ;;
    esac
}

sanitize_docker_tag_component() {
    local value
    value=$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | sed -E \
        -e 's/[^a-z0-9_.-]+/-/g' \
        -e 's/[-_.]+/-/g' \
        -e 's/^-+//' \
        -e 's/-+$//')
    if [[ -z "${value}" ]]; then
        value="detached"
    fi
    printf '%.48s\n' "${value}"
}

JD_CI_SKIP_MOONCAKE_BUILD="${JD_CI_SKIP_MOONCAKE_BUILD:-0}"
JD_CI_SKIP_SGL_KERNEL_BUILD="${JD_CI_SKIP_SGL_KERNEL_BUILD:-0}"
JD_CI_SKIP_TEST="${JD_CI_SKIP_TEST:-0}"

if [[ $# -gt 1 ]]; then
    usage_error "参数只能选择一个执行模式"
fi

case "${1:--r}" in
    -r|--review|note__merge_request)
        CI_MODE="review"
        EVENT_TYPE="note__merge_request"
        RUN_CI_TESTS=1
        PUBLISH_IMAGES=0
        ;;
    -m|--merge|merge_request__merged)
        CI_MODE="merge"
        EVENT_TYPE="merge_request__merged"
        RUN_CI_TESTS=0
        PUBLISH_IMAGES=1
        ;;
    -t|--temp-image)
        CI_MODE="temp-image"
        EVENT_TYPE="temp_image"
        RUN_CI_TESTS=1
        PUBLISH_IMAGES=1
        ;;
    -h|--help)
        print_usage
        exit 0
        ;;
    *)
        usage_error "未知参数: ${1}"
        ;;
esac

validate_binary_option "JD_CI_SKIP_SGL_KERNEL_BUILD" "${JD_CI_SKIP_SGL_KERNEL_BUILD}"
validate_binary_option "JD_CI_SKIP_MOONCAKE_BUILD" "${JD_CI_SKIP_MOONCAKE_BUILD}"
validate_binary_option "JD_CI_SKIP_TEST" "${JD_CI_SKIP_TEST}"

if [[ "${CI_MODE}" != "temp-image" ]] \
    && [[ "${JD_CI_SKIP_SGL_KERNEL_BUILD}" == "1" \
        || "${JD_CI_SKIP_MOONCAKE_BUILD}" == "1" \
        || "${JD_CI_SKIP_TEST}" == "1" ]]; then
    usage_error "组件和测试跳过选项仅允许在 -t / --temp-image 模式使用"
fi

if [[ "${CI_MODE}" == "temp-image" && "${JD_CI_SKIP_TEST}" == "1" ]]; then
    RUN_CI_TESTS=0
fi

CI_WORK_DIR="${CI_WORK_DIR:-/export/zhangyu}"
CI_USER="xn_testdev_ci"
CI_USER_SSH_DIR="/root/.ssh"

echo "[SGLang CI] CI_MODE=${CI_MODE}"
echo "[SGLang CI] EVENT_TYPE=${EVENT_TYPE}"
echo "[SGLang CI] RUN_CI_TESTS=${RUN_CI_TESTS}"
echo "[SGLang CI] PUBLISH_IMAGES=${PUBLISH_IMAGES}"

CI_SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_PATH="$(cd "${CI_SCRIPT_PATH}/../.." && pwd)"
echo "[SGLang CI] SOURCE_PATH=${SOURCE_PATH}"

require_clean_worktree() {
    local source_path="$1"
    local git_status_output

    if ! git -C "${source_path}" diff --quiet --exit-code; then
        echo "[SGLang CI] ERROR: 工作区不干净，git diff 非空；请先提交或清理改动后再运行 CI。" >&2
        echo "[SGLang CI] git diff --stat:" >&2
        git -C "${source_path}" diff --stat >&2 || true
        return 1
    fi

    git_status_output="$(git -C "${source_path}" status --porcelain)"
    if [[ -n "${git_status_output}" ]]; then
        echo "[SGLang CI] ERROR: 工作区不干净，git status 非空；请先提交或清理改动后再运行 CI。" >&2
        echo "[SGLang CI] git status --porcelain:" >&2
        echo "${git_status_output}" >&2
        return 1
    fi
}

require_clean_worktree "${SOURCE_PATH}"

pushd "${SOURCE_PATH}" > /dev/null
COMMIT_SHA=$(git rev-parse HEAD)
COMMIT_ID="${COMMIT_SHA:0:9}"
BRANCH_NAME=$(git branch --show-current)
if [[ -z "${BRANCH_NAME}" ]]; then
    BRANCH_NAME=$(git rev-parse --abbrev-ref HEAD)
fi
popd > /dev/null
echo "[SGLang CI] COMMIT_ID=${COMMIT_ID}"
echo "[SGLang CI] COMMIT_SHA=${COMMIT_SHA}"
echo "[SGLang CI] BRANCH_NAME=${BRANCH_NAME}"

BRANCH_NAME_FOR_DOCKER=$(sanitize_docker_tag_component "${BRANCH_NAME}")
BASE_IMAGE_PREFIX="images-infra-cn-east-1-inner.jcr.service.jdcloud.com/lmsysorg/sglang"

pushd "${SOURCE_PATH}" && BASE_IMAGE_TAG=$(git describe --tags --match 'v[0-9]*' --abbrev=0 HEAD) && popd
echo "[SGLang CI] BASE_IMAGE_TAG=${BASE_IMAGE_TAG}"

RELEASE_ARTIFACT_BRANCH="JD-${BASE_IMAGE_TAG}"
if [[ "${CI_MODE}" == "temp-image" && "${BRANCH_NAME}" == "${RELEASE_ARTIFACT_BRANCH}" ]]; then
    usage_error "-t / --temp-image 不能在正式分支 ${RELEASE_ARTIFACT_BRANCH} 运行，请使用 -r 或 -m"
fi

BASE_IMAGE="${BASE_IMAGE_PREFIX}:${BASE_IMAGE_TAG}"
CI_ARTIFACT_ROOT="${CI_WORK_DIR}/ci/sglang/jd-ci"
CI_RUNNER_ID="${COMMIT_ID}"
CI_RUN_ID="${CI_RUNNER_ID}"
CI_RUNNER_ROOT="${CI_ARTIFACT_ROOT}/runners/${CI_RUNNER_ID}"
CI_FINAL_STATE_ID="${CI_RUNNER_ID}-$$-${RANDOM}"
CI_FINAL_STATE_ROOT="${CI_ARTIFACT_ROOT}/final-state/${CI_FINAL_STATE_ID}"

TIMESTAMP=$(date +%Y%m%d%H%M%S)
CONTAINER_NAME="SGLANG_CI_BASE_IMAGE_${BASE_IMAGE_TAG}-BRANCH_${BRANCH_NAME_FOR_DOCKER}-COMMIT_${COMMIT_ID}-RUN_${CI_RUN_ID}"
CONTAINER_NAME=${CONTAINER_NAME,,} # 转小写
echo "[SGLang CI] CONTAINER_NAME=${CONTAINER_NAME}"

MOONCAKE_VERSION=$(docker run --rm --entrypoint bash "${BASE_IMAGE}" -c "pip list 2>/dev/null | grep -i '^mooncake-transfer-engine' | awk '{print \$2}' | head -1")
MOONCAKE_VERSION_TAG="v${MOONCAKE_VERSION}"

CLOUD_IMAGE="images-infra-cn-east-1-inner.jcr.service.jdcloud.com/sglang:${BASE_IMAGE_TAG}_JD_${COMMIT_ID}"
CLOUD_IMAGE=${CLOUD_IMAGE,,} # 转小写

echo "[SGLang CI] CLOUD_IMAGE:${CLOUD_IMAGE}"

# sgl-kernel wheel 持久化缓存目录。容器内 env/build_sgl_kernel.sh 会继续按
# CUDA Toolkit 版本和目标架构分桶，避免 H-only wheel 被 B 系列复用。
SGLANG_KERNEL_TARGET_ARCHS="${SGLANG_KERNEL_TARGET_ARCHS:-sm90,sm90a,sm100a,sm103a}"
JD_CI_BASE_REF="${JD_CI_BASE_REF:-${BASE_IMAGE_TAG}}"
JD_CI_SERVER_API_GPU_ID="${JD_CI_SERVER_API_GPU_ID:-0}"
JD_CI_SERVER_API_MODEL_PATH="${JD_CI_SERVER_API_MODEL_PATH:-/mnt/nas/models/Qwen2.5-VL-7B-Instruct/}"
JD_CI_SERVER_API_TIMEOUT_SEC="${JD_CI_SERVER_API_TIMEOUT_SEC:-600}"
JD_CI_OPERATOR_AVAILABLE_GPUS="${JD_CI_OPERATOR_AVAILABLE_GPUS:-}"
JD_CI_DUMP_LOGS_ON_FAILURE="${JD_CI_DUMP_LOGS_ON_FAILURE:-1}"
JD_CI_DUMP_FULL_LOGS="${JD_CI_DUMP_FULL_LOGS:-1}"
JD_CI_DUMP_LOG_MAX_BYTES="${JD_CI_DUMP_LOG_MAX_BYTES:-0}"
JD_CI_TEMP_ARTIFACT_ROOT=""
PERSISTENT_SGL_KERNEL_CACHE_HOST="${CI_WORK_DIR}/ci/sglang/sgl-kernel/${BASE_IMAGE_TAG}"
PERSISTENT_MOONCAKE_ENGINE_CACHE_HOST="${CI_WORK_DIR}/ci/sglang/mooncake_te/${MOONCAKE_VERSION_TAG}"
case "${CI_MODE}" in
    review|merge)
        JD_CI_ARTIFACT_SCOPE="persistent"
        if [[ "${CI_MODE}" == "review" ]]; then
            SGL_KERNEL_ARTIFACT_SCOPE="persistent-build"
            MOONCAKE_ARTIFACT_SCOPE="persistent-build"
        else
            SGL_KERNEL_ARTIFACT_SCOPE="persistent-cache"
            MOONCAKE_ARTIFACT_SCOPE="persistent-cache"
        fi
        WHEEL_CACHE_HOST="${PERSISTENT_SGL_KERNEL_CACHE_HOST}"
        MOONCAKE_ENGINE_CACHE_HOST="${PERSISTENT_MOONCAKE_ENGINE_CACHE_HOST}"
        ;;
    temp-image)
        JD_CI_ARTIFACT_SCOPE="temporary"
        JD_CI_TEMP_ARTIFACT_ROOT="${CI_RUNNER_ROOT}/artifacts"
        WHEEL_CACHE_HOST="${JD_CI_TEMP_ARTIFACT_ROOT}/sgl-kernel/${BASE_IMAGE_TAG}"
        MOONCAKE_ENGINE_CACHE_HOST="${JD_CI_TEMP_ARTIFACT_ROOT}/mooncake_te/${MOONCAKE_VERSION_TAG}"
        if [[ "${JD_CI_SKIP_SGL_KERNEL_BUILD}" == "1" ]]; then
            SGL_KERNEL_ARTIFACT_SCOPE="base-image"
        else
            SGL_KERNEL_ARTIFACT_SCOPE="temporary-build"
        fi
        if [[ "${JD_CI_SKIP_MOONCAKE_BUILD}" == "1" ]]; then
            MOONCAKE_ARTIFACT_SCOPE="base-image"
        else
            MOONCAKE_ARTIFACT_SCOPE="temporary-build"
        fi
        ;;
esac
echo "[SGLang CI] SGLANG_KERNEL_TARGET_ARCHS=${SGLANG_KERNEL_TARGET_ARCHS}"
echo "[SGLang CI] RELEASE_ARTIFACT_BRANCH=${RELEASE_ARTIFACT_BRANCH}"
echo "[SGLang CI] JD_CI_ARTIFACT_SCOPE=${JD_CI_ARTIFACT_SCOPE}"
echo "[SGLang CI] SGL_KERNEL_ARTIFACT_SCOPE=${SGL_KERNEL_ARTIFACT_SCOPE}"
echo "[SGLang CI] MOONCAKE_ARTIFACT_SCOPE=${MOONCAKE_ARTIFACT_SCOPE}"
if [[ -n "${JD_CI_TEMP_ARTIFACT_ROOT}" ]]; then
    echo "[SGLang CI] JD_CI_TEMP_ARTIFACT_ROOT=${JD_CI_TEMP_ARTIFACT_ROOT}"
fi
echo "[SGLang CI] WHEEL_CACHE_BASE=${WHEEL_CACHE_HOST}"
echo "[SGLang CI] MOONCAKE_VERSION_TAG=${MOONCAKE_VERSION_TAG}"
echo "[SGLang CI] MOONCAKE_TE_CACHE_BASE=${MOONCAKE_ENGINE_CACHE_HOST}"

if [[ "${CI_MODE}" == "merge" ]]; then
    MOONCAKE_FORCE_REBUILD=0
    MOONCAKE_REQUIRE_CACHE=1
else
    MOONCAKE_FORCE_REBUILD=1
    MOONCAKE_REQUIRE_CACHE=0
fi
echo "[SGLang CI] JD_CI_SKIP_MOONCAKE_BUILD=${JD_CI_SKIP_MOONCAKE_BUILD}"
echo "[SGLang CI] JD_CI_SKIP_SGL_KERNEL_BUILD=${JD_CI_SKIP_SGL_KERNEL_BUILD}"
echo "[SGLang CI] JD_CI_BASE_REF=${JD_CI_BASE_REF}"
echo "[SGLang CI] JD_CI_SERVER_API_MODEL_PATH=${JD_CI_SERVER_API_MODEL_PATH}"
echo "[SGLang CI] JD_CI_DUMP_LOGS_ON_FAILURE=${JD_CI_DUMP_LOGS_ON_FAILURE}"
echo "[SGLang CI] JD_CI_DUMP_FULL_LOGS=${JD_CI_DUMP_FULL_LOGS}"
echo "[SGLang CI] JD_CI_DUMP_LOG_MAX_BYTES=${JD_CI_DUMP_LOG_MAX_BYTES}"
echo "[SGLang CI] MOONCAKE_FORCE_REBUILD=${MOONCAKE_FORCE_REBUILD}"
echo "[SGLang CI] MOONCAKE_REQUIRE_CACHE=${MOONCAKE_REQUIRE_CACHE}"

CI_LOGS_DIR="${CI_RUNNER_ROOT}/logs"
CI_CONTAINER_LOGS_DIR="${CI_LOGS_DIR}/containers"
CI_BUILD_LOGS_DIR="${CI_LOGS_DIR}/builds"
CI_TEST_LOGS_DIR="${CI_LOGS_DIR}/tests"
MAIN_PIPELINE_LOG="${CI_CONTAINER_LOGS_DIR}/sglang.log"
SGL_KERNEL_BUILD_LOG="${CI_BUILD_LOGS_DIR}/sgl-kernel.log"
MOONCAKE_TE_BUILD_LOG="${CI_BUILD_LOGS_DIR}/mooncake-te.log"
FINAL_MAIN_TAIL_LOG="${CI_FINAL_STATE_ROOT}/containers/sglang.log"

CI_RUNNER_WORK_DIR="${CI_RUNNER_ROOT}/work"
MAIN_CONTAINER_WORK_DIR="${CI_RUNNER_WORK_DIR}/containers/sglang"
MAIN_CONTAINER_TMP_DIR="${MAIN_CONTAINER_WORK_DIR}/tmp"
SGL_KERNEL_WORK_DIR="${CI_RUNNER_WORK_DIR}/builds/sgl-kernel"
MOONCAKE_TE_WORK_DIR="${CI_RUNNER_WORK_DIR}/builds/mooncake-te"
CPU_MOCK_TEST_WORK_DIR="${CI_RUNNER_WORK_DIR}/tests/cpu-mock"
SERVER_API_TEST_WORK_DIR="${CI_RUNNER_WORK_DIR}/tests/server-api"
OPERATOR_TEST_WORK_DIR="${CI_RUNNER_WORK_DIR}/tests/operator"

CI_RUNNER_CLEANUP_TIMEOUT_SEC="${CI_RUNNER_CLEANUP_TIMEOUT_SEC:-300}"
DOCKER_CLEANUP_TIMEOUT_SEC="${DOCKER_CLEANUP_TIMEOUT_SEC:-60}"
ACTIVE_DOCKER_PID=""
FAILURE_LOGS_DUMPED=0
FINAL_FAILURE_SUMMARY=""
MAIN_EXIT_CODE=""

run_with_timeout() {
    local seconds="$1"
    shift
    if command -v timeout >/dev/null 2>&1; then
        timeout "${seconds}s" "$@"
    else
        "$@"
    fi
}

capture_recent_output() {
    local output_file="$1"
    local max_lines="${2:-200}"
    awk -v output_file="${output_file}" -v max_lines="${max_lines}" '
        {
            print
            fflush()
            recent[NR % max_lines] = $0
        }
        END {
            count = NR < max_lines ? NR : max_lines
            start = NR - count + 1
            for (line_number = start; line_number <= NR; line_number++) {
                print recent[line_number % max_lines] > output_file
            }
            close(output_file)
        }
    '
}

rm_ci_runner_dir() {
    run_with_timeout "${CI_RUNNER_CLEANUP_TIMEOUT_SEC}" rm -rf "${CI_RUNNER_ROOT}" 2>/dev/null || true
}

cleanup_ci_runner_dir() {
    local label="${1:-收尾清理}"
    if [[ ! "${CI_RUNNER_ID}" =~ ^[0-9a-f]{9}$ ]] \
        || [[ "${CI_RUNNER_ROOT:-}" != "${CI_ARTIFACT_ROOT}/runners/${CI_RUNNER_ID}" ]]; then
        echo "[SGLang CI] WARN: 拒绝清理非预期 runner 目录: ${CI_RUNNER_ROOT:-<unset>}"
        return 1
    fi
    rm_ci_runner_dir
    if [[ -d "${CI_RUNNER_ROOT}" ]]; then
        echo "[SGLang CI] 普通删除未清空 runner 目录，尝试容器内 root 兜底清理: ${CI_RUNNER_ROOT}"
        run_with_timeout "${DOCKER_CLEANUP_TIMEOUT_SEC}" docker run --rm \
            --platform linux/amd64 \
            -v "${CI_RUNNER_ROOT}:/ci-runner" \
            --entrypoint /bin/bash \
            "${BASE_IMAGE}" \
            -c "shopt -s dotglob nullglob; rm -rf /ci-runner/*" >/dev/null 2>&1 || true
        rm_ci_runner_dir
    fi
    if [[ -d "${CI_RUNNER_ROOT}" ]]; then
        echo "[SGLang CI] WARN: ${label} runner 目录未完全删除: ${CI_RUNNER_ROOT}"
        return 1
    else
        echo "[SGLang CI] ${label} runner 目录已删除: ${CI_RUNNER_ROOT}"
    fi
}

cleanup_ci_final_state_dir() {
    if [[ ! "${CI_FINAL_STATE_ID}" =~ ^[0-9a-f]{9}-[0-9]+-[0-9]+$ ]] \
        || [[ "${CI_FINAL_STATE_ROOT:-}" != "${CI_ARTIFACT_ROOT}/final-state/${CI_FINAL_STATE_ID}" ]]; then
        echo "[SGLang CI] WARN: 拒绝清理非预期 final-state 目录: ${CI_FINAL_STATE_ROOT:-<unset>}"
        return 1
    fi
    run_with_timeout "${CI_RUNNER_CLEANUP_TIMEOUT_SEC}" \
        rm -rf "${CI_FINAL_STATE_ROOT}" 2>/dev/null || true
    if [[ -d "${CI_FINAL_STATE_ROOT}" ]]; then
        echo "[SGLang CI] WARN: final-state 目录未完全删除: ${CI_FINAL_STATE_ROOT}"
        return 1
    fi
}

cleanup_container_by_name() {
    local name="$1"
    local label="$2"
    if [[ -z "${name}" ]]; then
        return
    fi
    echo "[SGLang CI] 清理${label}(若存在): ${name}"
    run_with_timeout "${DOCKER_CLEANUP_TIMEOUT_SEC}" docker rm -f "${name}" >/dev/null 2>&1 || true
}

stop_active_docker_cli() {
    if [[ -n "${ACTIVE_DOCKER_PID}" ]] && kill -0 "${ACTIVE_DOCKER_PID}" 2>/dev/null; then
        echo "[SGLang CI] 停止当前 docker CLI 进程: ${ACTIVE_DOCKER_PID}"
        kill -TERM "${ACTIVE_DOCKER_PID}" 2>/dev/null || true
        for _ in {1..10}; do
            kill -0 "${ACTIVE_DOCKER_PID}" 2>/dev/null || break
            sleep 1
        done
        kill -KILL "${ACTIVE_DOCKER_PID}" 2>/dev/null || true
        wait "${ACTIVE_DOCKER_PID}" 2>/dev/null || true
    fi
    ACTIVE_DOCKER_PID=""
}

run_docker_attached() {
    "$@" &
    ACTIVE_DOCKER_PID=$!
    wait "${ACTIVE_DOCKER_PID}"
    local status=$?
    ACTIVE_DOCKER_PID=""
    return ${status}
}

cleanup_on_exit() {
    local status=$?
    set +e
    trap - EXIT INT TERM
    stop_active_docker_cli
    cleanup_container_by_name "${CONTAINER_NAME}" "主容器"
    if [[ ${status} -ne 0 ]]; then
        capture_ci_failure_summary "${status}"
    fi
    if [[ ${status} -ne 0 && ${FAILURE_LOGS_DUMPED} -eq 0 ]]; then
        dump_ci_failure_logs
    fi
    if ! cleanup_ci_runner_dir "收尾清理"; then
        echo "[SGLang CI] ERROR: runner 目录清理失败"
        if [[ ${status} -eq 0 ]]; then
            status=1
            FINAL_FAILURE_SUMMARY="[JD CI FAILURE] overall_exit_code=1
[JD CI FAILURE] root_cause=runner 目录清理失败: ${CI_RUNNER_ROOT}"
        fi
    fi
    if ! cleanup_ci_final_state_dir; then
        echo "[SGLang CI] ERROR: final-state 目录清理失败"
        if [[ ${status} -eq 0 ]]; then
            status=1
            FINAL_FAILURE_SUMMARY="[JD CI FAILURE] overall_exit_code=1
[JD CI FAILURE] root_cause=final-state 目录清理失败: ${CI_FINAL_STATE_ROOT}"
        else
            FINAL_FAILURE_SUMMARY+=$'\n'"[JD CI FAILURE] cleanup_error=final-state 目录清理失败: ${CI_FINAL_STATE_ROOT}"
        fi
    fi
    if [[ ${status} -ne 0 ]]; then
        print_final_failure_summary "${status}"
    fi
    return ${status}
}

capture_ci_failure_summary() {
    local status="$1"
    local summary_args=(
        --logs-dir "${CI_LOGS_DIR}"
        --failure-summary
        --overall-exit-code "${status}"
        --fallback-logs-dir "${CI_FINAL_STATE_ROOT}"
    )
    if [[ -n "${MAIN_EXIT_CODE}" ]]; then
        summary_args+=(--main-exit-code "${MAIN_EXIT_CODE}")
    fi
    if command -v python3 >/dev/null 2>&1; then
        FINAL_FAILURE_SUMMARY="$(
            python3 "${SOURCE_PATH}/test/jd-ci/report/dump_ci_logs.py" \
                "${summary_args[@]}" 2>&1
        )" || true
    fi
    if [[ -z "${FINAL_FAILURE_SUMMARY}" ]]; then
        FINAL_FAILURE_SUMMARY="[JD CI FAILURE] overall_exit_code=${status}
[JD CI FAILURE] root_cause=失败摘要生成失败，请查看上方完整日志转储"
    fi
}

print_final_failure_summary() {
    local status="$1"
    echo "[SGLang CI] ========================================"
    echo "[SGLang CI]  最终失败原因（现场清理已完成）"
    echo "[SGLang CI] ========================================"
    printf '%s\n' "${FINAL_FAILURE_SUMMARY}"
    echo "[SGLang CI] FINAL_STATUS=FAILED exit_code=${status}"
}

dump_ci_failure_logs() {
    FAILURE_LOGS_DUMPED=1
    if [[ "${JD_CI_DUMP_LOGS_ON_FAILURE}" != "1" ]]; then
        echo "[SGLang CI] 跳过失败日志转储 (JD_CI_DUMP_LOGS_ON_FAILURE=${JD_CI_DUMP_LOGS_ON_FAILURE})"
        return
    fi
    if [[ ! -d "${CI_LOGS_DIR}" ]]; then
        echo "[SGLang CI] 失败日志目录不存在，无法转储: ${CI_LOGS_DIR}"
        return
    fi

    echo "[SGLang CI] ========================================"
    echo "[SGLang CI]  失败日志转储到流水线 stdout"
    echo "[SGLang CI] ========================================"
    local dump_args=(--logs-dir "${CI_LOGS_DIR}" --max-bytes "${JD_CI_DUMP_LOG_MAX_BYTES}")
    if [[ "${JD_CI_DUMP_FULL_LOGS}" == "1" ]]; then
        dump_args+=(--full)
    fi
    if command -v python3 >/dev/null 2>&1; then
        python3 "${SOURCE_PATH}/test/jd-ci/report/dump_ci_logs.py" "${dump_args[@]}" || true
    else
        local dump_full_arg=""
        if [[ "${JD_CI_DUMP_FULL_LOGS}" == "1" ]]; then
            dump_full_arg="--full"
        fi
        echo "[SGLang CI] 宿主机 python3 不存在，使用基础镜像转储失败日志"
        docker run --rm \
            --platform linux/amd64 \
            -v "${SOURCE_PATH}:${SOURCE_PATH}" \
            -v "${CI_RUNNER_ROOT}:${CI_RUNNER_ROOT}:ro" \
            -w "${SOURCE_PATH}" \
            --entrypoint /bin/bash \
            "${BASE_IMAGE}" \
            -c "python3 '${SOURCE_PATH}/test/jd-ci/report/dump_ci_logs.py' --logs-dir '${CI_LOGS_DIR}' --max-bytes '${JD_CI_DUMP_LOG_MAX_BYTES}' ${dump_full_arg}" \
            || true
    fi
}

trap cleanup_on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cleanup_ci_runner_dir "启动前清理"
mkdir -p \
    "${CI_ARTIFACT_ROOT}" \
    "${CI_LOGS_DIR}" \
    "${CI_CONTAINER_LOGS_DIR}" \
    "${CI_BUILD_LOGS_DIR}" \
    "${CI_TEST_LOGS_DIR}" \
    "${CI_FINAL_STATE_ROOT}/containers" \
    "${MAIN_CONTAINER_TMP_DIR}" \
    "${SGL_KERNEL_WORK_DIR}" \
    "${MOONCAKE_TE_WORK_DIR}/compile" \
    "${CPU_MOCK_TEST_WORK_DIR}" \
    "${SERVER_API_TEST_WORK_DIR}" \
    "${OPERATOR_TEST_WORK_DIR}" \
    "${WHEEL_CACHE_HOST}" \
    "${MOONCAKE_ENGINE_CACHE_HOST}"
chmod 1777 "${MAIN_CONTAINER_TMP_DIR}" 2>/dev/null || true
echo "[SGLang CI] CI_RUNNER_ID=${CI_RUNNER_ID}"
echo "[SGLang CI] CI_RUNNER_ROOT=${CI_RUNNER_ROOT}"
echo "[SGLang CI] CI_LOGS_DIR=${CI_LOGS_DIR}"
echo "[SGLang CI] MAIN_PIPELINE_LOG=${MAIN_PIPELINE_LOG}"
echo "[SGLang CI] SGL_KERNEL_BUILD_LOG=${SGL_KERNEL_BUILD_LOG}"
echo "[SGLang CI] MOONCAKE_TE_BUILD_LOG=${MOONCAKE_TE_BUILD_LOG}"
echo "[SGLang CI] CI_RUNNER_WORK_DIR=${CI_RUNNER_WORK_DIR}"

# 构建镜像信息
IMAGE_ENTRYPOINT_SCRIPT="${MAIN_CONTAINER_WORK_DIR}/entrypoint.sh"
pushd "${SOURCE_PATH}" > /dev/null
GIT_DIFF_MSGS=$(git log "${BASE_IMAGE_TAG}..HEAD" --oneline --pretty=format:"%h | %an | %ad | %s" --date=short)
popd > /dev/null
echo "[SGLang CI] GIT_DIFF_MSGS:"
echo "${GIT_DIFF_MSGS}"

cat > "${IMAGE_ENTRYPOINT_SCRIPT}" <<EOF
#!/bin/bash
set -e

cat <<'BANNER'
====================================================================================================
 JD SGLang 镜像 ${CLOUD_IMAGE} 信息
----------------------------------------------------------------------------------------------------
 基础镜像 : ${BASE_IMAGE}
 当前分支 : ${BRANCH_NAME}
 最新提交 : ${COMMIT_ID}
 构建时间 : ${TIMESTAMP}
----------------------------------------------------------------------------------------------------
 相较于 ${BASE_IMAGE_TAG} 分支的所有历史改动:
${GIT_DIFF_MSGS}
====================================================================================================
BANNER

if [ -f /sgl-workspace/setup_cuda_compat.sh ]; then
    source /sgl-workspace/setup_cuda_compat.sh
fi

exec "\$@"
EOF
chmod +x "${IMAGE_ENTRYPOINT_SCRIPT}"

# 启动容器并运行单测，捕获状态码
set +e

cleanup_container_by_name "${CONTAINER_NAME}" "启动前主容器"

run_docker_attached docker run \
    --name "${CONTAINER_NAME}" \
    --platform linux/amd64 \
    --net=host --pid=host --ipc=host --privileged \
    --gpus all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e DISPLAY=${DISPLAY:-} \
    -e WHEEL_CACHE_DIR=/wheels \
    -e MOONCAKE_ENGINE_CACHE_DIR="${MOONCAKE_ENGINE_CACHE_HOST}" \
    -e MOONCAKE_WHEEL_CACHE_DIR="${MOONCAKE_ENGINE_CACHE_HOST}" \
    -e MOONCAKE_FORCE_REBUILD="${MOONCAKE_FORCE_REBUILD}" \
    -e MOONCAKE_REQUIRE_CACHE="${MOONCAKE_REQUIRE_CACHE}" \
    -e BASE_IMAGE_TAG="${BASE_IMAGE_TAG}" \
    -e EVENT_TYPE="${EVENT_TYPE}" \
    -e SGLANG_KERNEL_TARGET_ARCHS="${SGLANG_KERNEL_TARGET_ARCHS}" \
    -e JD_LOG_DIR="${CI_LOGS_DIR}/helpers/sglang" \
    -e JD_CI_BASE_REF="${JD_CI_BASE_REF}" \
    -e JD_CI_SERVER_API_GPU_ID="${JD_CI_SERVER_API_GPU_ID}" \
    -e JD_CI_SERVER_API_MODEL_PATH="${JD_CI_SERVER_API_MODEL_PATH}" \
    -e JD_CI_SERVER_API_TIMEOUT_SEC="${JD_CI_SERVER_API_TIMEOUT_SEC}" \
    -e JD_CI_OPERATOR_AVAILABLE_GPUS="${JD_CI_OPERATOR_AVAILABLE_GPUS}" \
    -e CI_TMP_DIR="${MAIN_CONTAINER_WORK_DIR}" \
    -e TMPDIR="${MAIN_CONTAINER_TMP_DIR}" \
    -e TMP="${MAIN_CONTAINER_TMP_DIR}" \
    -e TEMP="${MAIN_CONTAINER_TMP_DIR}" \
    -e CUDA_CACHE_PATH="${MAIN_CONTAINER_WORK_DIR}/cuda-cache" \
    -e SGL_KERNEL_BUILD_TMPDIR="${SGL_KERNEL_WORK_DIR}/tmp" \
    -e MOONCAKE_TMP_DIR="${MOONCAKE_TE_WORK_DIR}/tmp" \
    -e UV_CACHE_DIR="${MAIN_CONTAINER_WORK_DIR}/uv-cache" \
    -e PIP_CACHE_DIR="${MAIN_CONTAINER_WORK_DIR}/pip-cache" \
    -e XDG_CACHE_HOME="${MAIN_CONTAINER_WORK_DIR}/xdg-cache" \
    -e TORCH_EXTENSIONS_DIR="${MAIN_CONTAINER_WORK_DIR}/torch-extensions" \
    -e TRITON_CACHE_DIR="${MAIN_CONTAINER_WORK_DIR}/triton-cache" \
    -e TORCHINDUCTOR_CACHE_DIR="${MAIN_CONTAINER_WORK_DIR}/torchinductor-cache" \
    -e GIT_SSH_COMMAND='ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null' \
    -v "${CI_USER_SSH_DIR}:/root/.ssh:ro" \
    -v /sys/kernel/debug:/sys/kernel/debug \
    -v /etc/apt/sources.list:/etc/apt/sources.list \
    -v "${MAIN_CONTAINER_TMP_DIR}:/tmp" \
    -v /mnt/nas/:/mnt/nas/ \
    -v "${CI_WORK_DIR}:${CI_WORK_DIR}" \
    -v "${WHEEL_CACHE_HOST}:/wheels" \
    -v "${SOURCE_PATH}:${SOURCE_PATH}" \
    -w "${SOURCE_PATH}" \
    --entrypoint /bin/bash \
    "${BASE_IMAGE}" \
    -c "
        set -euo pipefail
        echo '[JD CI] CI 容器启动成功!'
        echo '[JD CI] CI 工作路径:' \$(pwd)

        run_with_isolated_workspace() {
            local workspace=\"\$1\"
            shift
            mkdir -p \
                \"\${workspace}/tmp\" \
                \"\${workspace}/cuda-cache\" \
                \"\${workspace}/uv-cache\" \
                \"\${workspace}/pip-cache\" \
                \"\${workspace}/xdg-cache\" \
                \"\${workspace}/torch-extensions\" \
                \"\${workspace}/triton-cache\" \
                \"\${workspace}/torchinductor-cache\"
            env \
                TMPDIR=\"\${workspace}/tmp\" \
                TMP=\"\${workspace}/tmp\" \
                TEMP=\"\${workspace}/tmp\" \
                CUDA_CACHE_PATH=\"\${workspace}/cuda-cache\" \
                UV_CACHE_DIR=\"\${workspace}/uv-cache\" \
                PIP_CACHE_DIR=\"\${workspace}/pip-cache\" \
                XDG_CACHE_HOME=\"\${workspace}/xdg-cache\" \
                TORCH_EXTENSIONS_DIR=\"\${workspace}/torch-extensions\" \
                TRITON_CACHE_DIR=\"\${workspace}/triton-cache\" \
                TORCHINDUCTOR_CACHE_DIR=\"\${workspace}/torchinductor-cache\" \
                \"\$@\"
        }

        mkdir -p \
            '${MAIN_CONTAINER_TMP_DIR}' \
            '${SGL_KERNEL_WORK_DIR}' \
            '${MOONCAKE_TE_WORK_DIR}' \
            '${CPU_MOCK_TEST_WORK_DIR}' \
            '${SERVER_API_TEST_WORK_DIR}' \
            '${OPERATOR_TEST_WORK_DIR}'
        chmod 1777 '${MAIN_CONTAINER_TMP_DIR}' 2>/dev/null || true

        # ---------- 共享环境 ----------
        source ${SOURCE_PATH}/test/jd-ci/env/setup_env.sh ${SOURCE_PATH} ${BASE_IMAGE_TAG}
        mkdir -p /sgl-workspace
        cp '${IMAGE_ENTRYPOINT_SCRIPT}' /sgl-workspace/entrypoint.sh
        chmod +x /sgl-workspace/entrypoint.sh
        bash '${SOURCE_PATH}/test/jd-ci/image/install_dspark_sps_table.sh' \
            '${SOURCE_PATH}/deepseek-v4-flash-0731-sps-2d-1p-4gpu-1d-4gpu-g2-flashinfer-cutlass.json' \
            '/deepseek-v4-flash-0731-sps-2d-1p-4gpu-1d-4gpu-g2-flashinfer-cutlass.json'

        # ---------- 编译 ----------

        if [[ '${JD_CI_SKIP_MOONCAKE_BUILD}' == '1' ]]; then
            echo '[JD CI] 跳过 mooncake 编译 (JD_CI_SKIP_MOONCAKE_BUILD=1)'
        else
            echo '[JD CI] 开始 mooncake 编译'
            echo '[JD CI] mooncake TE cache: ${MOONCAKE_ENGINE_CACHE_HOST}'
            echo '[JD CI] MOONCAKE_FORCE_REBUILD='\${MOONCAKE_FORCE_REBUILD}
            echo '[JD CI] MOONCAKE_REQUIRE_CACHE='\${MOONCAKE_REQUIRE_CACHE}
            set -o pipefail
            run_with_isolated_workspace '${MOONCAKE_TE_WORK_DIR}' \
                bash '${SOURCE_PATH}/test/jd-ci/env/build_mooncake.sh' \
                '${MOONCAKE_TE_WORK_DIR}/compile' 'te' \
                | tee '${MOONCAKE_TE_BUILD_LOG}'
            set +o pipefail
        fi

        if [[ '${JD_CI_SKIP_SGL_KERNEL_BUILD}' == '1' ]]; then
            echo '[JD CI] 跳过 sgl-kernel 编译 (JD_CI_SKIP_SGL_KERNEL_BUILD=1)'
        else
            echo '[JD CI] 开始 sgl-kernel 编译'
            set -o pipefail
            run_with_isolated_workspace '${SGL_KERNEL_WORK_DIR}' \
                env SGL_KERNEL_BUILD_TMPDIR='${SGL_KERNEL_WORK_DIR}/tmp' \
                bash '${SOURCE_PATH}/test/jd-ci/env/build_sgl_kernel.sh' \
                '${EVENT_TYPE}' \
                '${BASE_IMAGE_TAG}' \
                '${CI_WORK_DIR}' \
                '/wheels' \
                '/sgl-workspace/sglang/sgl-kernel' \
                '${PERSISTENT_SGL_KERNEL_CACHE_HOST}' 2>&1 | tee '${SGL_KERNEL_BUILD_LOG}'
            set +o pipefail
        fi

        if [[ '${RUN_CI_TESTS}' == '1' ]]; then
            jd_ci_failed=0

            echo '[JD CI] ========================================'
            echo '[JD CI]  CPU and Mock Regression: 固定累积 JD CPU/mock 回归（GPU 保持可见）'
            echo '[JD CI] ========================================'
            run_with_isolated_workspace '${CPU_MOCK_TEST_WORK_DIR}' \
                bash '${SOURCE_PATH}/test/jd-ci/pipeline/run_cpu_mock_regression.sh' \
                '${SOURCE_PATH}' '${JD_CI_BASE_REF}' '${CI_TEST_LOGS_DIR}' \
                || { echo '[JD CI] CPU and Mock Regression 失败'; jd_ci_failed=1; }

            echo '[JD CI] ========================================'
            echo '[JD CI]  Server and API Regression: 固定累积 JD Server/API 回归（dummy weight）'
            echo '[JD CI] ========================================'
            run_with_isolated_workspace '${SERVER_API_TEST_WORK_DIR}' \
                bash '${SOURCE_PATH}/test/jd-ci/pipeline/run_server_api_regression.sh' \
                '${SOURCE_PATH}' '${CI_TEST_LOGS_DIR}' \
                || { echo '[JD CI] Server and API Regression 失败'; jd_ci_failed=1; }

            echo '[JD CI] ========================================'
            echo '[JD CI]  Operator Correctness and Performance Regression: 固定累积 JD 算子 correctness + performance'
            echo '[JD CI] ========================================'
            run_with_isolated_workspace '${OPERATOR_TEST_WORK_DIR}' \
                bash '${SOURCE_PATH}/test/jd-ci/pipeline/run_operator_regression.sh' \
                '${SOURCE_PATH}' '${CI_TEST_LOGS_DIR}' \
                || { echo '[JD CI] Operator Correctness and Performance Regression 失败或资源不足'; jd_ci_failed=1; }

            echo '[JD CI] ========================================'
            echo '[JD CI]  三类固定累积 JD 回归汇总报告生成'
            echo '[JD CI] ========================================'
            python3 '${SOURCE_PATH}/test/jd-ci/report/generate_regression_summary.py' \
                --logs-dir '${CI_TEST_LOGS_DIR}' \
                --event-type '${EVENT_TYPE}' \
                --branch '${BRANCH_NAME}' \
                --commit '${COMMIT_ID}' \
                --base-image-tag '${BASE_IMAGE_TAG}' \
                || { echo '[JD CI] 汇总报告生成失败'; jd_ci_failed=1; }

            exit \${jd_ci_failed}
        elif [[ '${CI_MODE}' == 'temp-image' ]]; then
            echo '[JD CI] JD_CI_SKIP_TEST=1: 用户显式跳过临时镜像的全部 JD CI 回归'
            for test_area in cpu_mock server_api operator; do
                python3 '${SOURCE_PATH}/test/jd-ci/report/regression_report.py' \
                    --output '${CI_TEST_LOGS_DIR}/'\${test_area}'/report.json' \
                    --test-area "\${test_area}" \
                    --skip-reason 'JD_CI_SKIP_TEST=1 in temp-image mode'
            done
            python3 '${SOURCE_PATH}/test/jd-ci/report/generate_regression_summary.py' \
                --logs-dir '${CI_TEST_LOGS_DIR}' \
                --event-type '${EVENT_TYPE}' \
                --branch '${BRANCH_NAME}' \
                --commit '${COMMIT_ID}' \
                --base-image-tag '${BASE_IMAGE_TAG}'
        else
            echo '[JD CI] merge 模式: 任意分支跳过测试，只安装对应版本主分支的正式缓存并打包镜像'
        fi
    " 2>&1 | tee "${MAIN_PIPELINE_LOG}" \
        | capture_recent_output "${FINAL_MAIN_TAIL_LOG}"

MAIN_EXIT_CODE=${PIPESTATUS[0]}
set -e
EXIT_CODE=${MAIN_EXIT_CODE}
if [ ${EXIT_CODE} -ne 0 ]; then
    echo "[SGLang CI] 主 SGLang 流水线失败，退出码: ${MAIN_EXIT_CODE}"
fi

if [[ ${EXIT_CODE} -eq 0 && "${PUBLISH_IMAGES}" == "1" ]]; then
    echo "[SGLang CI] ${CI_MODE} 所有已启用门禁通过，开始生成 SGLang 镜像..."
    docker commit \
        -c "WORKDIR /sgl-workspace/sglang" \
        -c 'ENTRYPOINT ["/sgl-workspace/entrypoint.sh"]' \
        -c 'CMD ["/bin/bash"]' \
        -c "ENV LD_PRELOAD=/sgl-workspace/fake_dns.so" \
        -c "ENV MC_IB_PCI_RELAXED_ORDERING=1" \
        "${CONTAINER_NAME}" "${CLOUD_IMAGE}"
    docker run --rm --entrypoint /bin/bash "${CLOUD_IMAGE}" -lc \
        "echo 'f529eaa58c1ad62ddb5eacc8c40e2fbb13e8979e8f0c519c3aca33a67dc64752  /deepseek-v4-flash-0731-sps-2d-1p-4gpu-1d-4gpu-g2-flashinfer-cutlass.json' | sha256sum -c -"
    docker push "${CLOUD_IMAGE}"
    echo "[SGLang CI] SGLang 云上仓库镜像地址: ${CLOUD_IMAGE}"
elif [[ ${EXIT_CODE} -eq 0 ]]; then
    echo "[SGLang CI] review 模式完成，不创建或推送镜像。"
else
    echo "[SGLang CI] 流水线门禁失败，不创建或推送 SGLang 镜像。"
fi

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "[SGLang CI] 测试成功！"
else
    echo "[SGLang CI] 测试失败，退出码: ${EXIT_CODE}"
fi

if [ -f "${CI_TEST_LOGS_DIR}/jd_ci_report.md" ]; then
    echo "[SGLang CI] 汇总报告 Markdown: ${CI_TEST_LOGS_DIR}/jd_ci_report.md"
fi
if [ -f "${CI_TEST_LOGS_DIR}/jd_ci_report.json" ]; then
    echo "[SGLang CI] 汇总报告 JSON: ${CI_TEST_LOGS_DIR}/jd_ci_report.json"
fi

if [ ${EXIT_CODE} -ne 0 ]; then
    dump_ci_failure_logs
fi

exit ${EXIT_CODE}
