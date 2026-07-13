#!/bin/bash
# JD CI 共享环境初始化。由 run_jd_ci.sh / run_jd_p1_accuracy.sh / run_jd_p2_functional.sh 共用。
# 预期在 Docker 容器内执行，SOURCE_PATH 为 sglang 源码根目录，CI_WORK_DIR 为持久化工作目录。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env/setup_logger.sh
source "${SCRIPT_DIR}/setup_logger.sh" "jd_ci_env"

SOURCE_PATH="${1:?Usage: env/setup_env.sh <SOURCE_PATH> [BASE_IMAGE_TAG] [CI_WORK_DIR]}"
BASE_IMAGE_TAG="${2:-}"
CI_WORK_DIR="${3:-/home/zhangyu}"

jd_log_info "SOURCE_PATH=${SOURCE_PATH}"
jd_log_info "CI_WORK_DIR=${CI_WORK_DIR}"
jd_log_info "BASE_IMAGE_TAG=${BASE_IMAGE_TAG:-<not set>}"

# Fix git dubious ownership in mounted CI directories
git config --global --add safe.directory "${SOURCE_PATH}" 2>/dev/null || true

# ---------- pip 镜像 ----------
pip config set global.index-url https://mirrors.jd.com/pypi/web/simple 2>/dev/null || true
pip config set global.trusted-host mirrors.jd.com 2>/dev/null || true

# ---------- 引擎源码替换 ----------
if [ -d /sgl-workspace/sglang/python ]; then
    \cp -r "${SOURCE_PATH}/python/"* /sgl-workspace/sglang/python/
    jd_log_info "引擎源码替换完成"
else
    jd_log_info "WARNING: /sgl-workspace/sglang/python 不存在，跳过源码替换"
fi

# ---------- CUDA compat ----------
if [ -f "${SOURCE_PATH}/test/jd-ci/env/setup_cuda_compat.sh" ]; then
    mkdir -p /sgl-workspace
    \cp -f "${SOURCE_PATH}/test/jd-ci/env/setup_cuda_compat.sh" /sgl-workspace/setup_cuda_compat.sh
    chmod +x /sgl-workspace/setup_cuda_compat.sh
    source /sgl-workspace/setup_cuda_compat.sh
    sglang_ci_print_cuda_driver_probe
fi

# ---------- DeepGEMM 编译并行度 ----------
export SGLANG_JIT_DEEPGEMM_COMPILE_WORKERS="${SGLANG_JIT_DEEPGEMM_COMPILE_WORKERS:-64}"

# ---------- 缓存目录 ----------
if [ -n "${BASE_IMAGE_TAG}" ]; then
    export SGLANG_DG_CACHE_DIR="${CI_WORK_DIR}/ci/sglang/sgl-cache/${BASE_IMAGE_TAG}"
    jd_log_info "SGLANG_DG_CACHE_DIR=${SGLANG_DG_CACHE_DIR}"
fi

# ---------- fake_dns ----------
if [ -f "${SOURCE_PATH}/deploy/infer/fake_dns.so" ]; then
    \cp -f "${SOURCE_PATH}/deploy/infer/fake_dns.so" /sgl-workspace/fake_dns.so
    chmod +x /sgl-workspace/fake_dns.so
fi

# ---------- show_gids ----------
if [ -f "${SOURCE_PATH}/deploy/infer/show_gids" ]; then
    \cp -f "${SOURCE_PATH}/deploy/infer/show_gids" /usr/sbin/show_gids
    chmod +x /usr/sbin/show_gids
fi

# ---------- start.sh ----------
mkdir -p /sgl-workspace/sglang/starter/
if [ -f "${SOURCE_PATH}/deploy/infer/start.sh" ]; then
    \cp -f "${SOURCE_PATH}/deploy/infer/start.sh" /sgl-workspace/sglang/starter/start.sh
fi

# ---------- chat_template ----------
if [ -d "${SOURCE_PATH}/examples/chat_template" ]; then
    \cp -rf "${SOURCE_PATH}/examples/chat_template/" /sgl-workspace/sglang/examples/chat_template/
fi

# ---------- nvshmem ----------
if [ -d "${CI_WORK_DIR}/ci/nvshmem" ]; then
    rm -rf /usr/local/lib/python3.12/dist-packages/nvidia/nvshmem/* 2>/dev/null || true
    \cp -r "${CI_WORK_DIR}/ci/nvshmem/" /usr/local/lib/python3.12/dist-packages/nvidia/
    jd_log_info "nvshmem 替换完成"
fi

# ---------- PYTHONPATH ----------
export PYTHONPATH="${SOURCE_PATH}/python:${SOURCE_PATH}/test/jd-ci:${PYTHONPATH:-}"
jd_log_info "PYTHONPATH=${PYTHONPATH}"

jd_log_info "环境初始化完成"
