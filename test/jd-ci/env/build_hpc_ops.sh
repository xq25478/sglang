#!/bin/bash
set -euo pipefail

SOURCE_DIR="$1"
COMPILE_DIR="$2"
INSTALL_DIR="$3"
CACHE_DIR="$4"
FORCE_REBUILD="$5"
REQUIRE_CACHE="$6"

select_cached_wheel() {
    find "${CACHE_DIR}" \
        -maxdepth 1 \
        -type f \
        -name '*.whl' \
        -print |
        sort |
        tail -n 1
}

mkdir -p "${INSTALL_DIR}" "${CACHE_DIR}"
rm -rf "${INSTALL_DIR:?}/"*

WHEEL_FILE=""

# merge 模式：只使用已有正式缓存
if [[ "${REQUIRE_CACHE}" == "1" ]]; then
    WHEEL_FILE="$(select_cached_wheel)"

    if [[ -z "${WHEEL_FILE}" ]]; then
        echo "[JD CI] ERROR: 未找到 hpc-ops wheel 缓存: ${CACHE_DIR}" >&2
        exit 1
    fi

    echo "[JD CI] 使用 hpc-ops wheel 缓存: ${WHEEL_FILE}"
else
    if [[ ! -d "${SOURCE_DIR}" ]]; then
        echo "[JD CI] ERROR: hpc-ops 源码目录不存在: ${SOURCE_DIR}" >&2
        exit 1
    fi

    if [[ ! -f "${SOURCE_DIR}/Makefile" ]]; then
        echo "[JD CI] ERROR: hpc-ops 中未找到 Makefile: ${SOURCE_DIR}" >&2
        exit 1
    fi

    # 删除整个隔离构建目录，避免遗留隐藏文件和 CMakeCache
    rm -rf "${COMPILE_DIR}"
    mkdir -p "${COMPILE_DIR}"

    echo "[JD CI] 复制干净的 hpc-ops 源码"

    # 不复制任何历史构建产物
    tar \
        --exclude='./build' \
        --exclude='./dist' \
        --exclude='./_skbuild' \
        --exclude='./.pytest_cache' \
        --exclude='./.cache' \
        --exclude='./__pycache__' \
        --exclude='./*.egg-info' \
        -C "${SOURCE_DIR}" \
        -cf - . |
        tar -C "${COMPILE_DIR}" -xf -

    # 防御性清理嵌套目录中的旧 CMake 缓存
    find "${COMPILE_DIR}" \
        -type f \
        \( -name 'CMakeCache.txt' -o -name 'cmake_install.cmake' \) \
        -delete

    find "${COMPILE_DIR}" \
        -type d \
        \( -name 'CMakeFiles' -o -name '__pycache__' \) \
        -prune \
        -exec rm -rf {} +

    cd "${COMPILE_DIR}"

    echo "[JD CI] 在干净源码副本中执行 make wheel: ${COMPILE_DIR}"
    make wheel

    mapfile -t BUILT_WHEELS < <(
        find "${COMPILE_DIR}" \
            -type f \
            -path '*/dist/*.whl' \
            -print |
            sort
    )

    if (( ${#BUILT_WHEELS[@]} == 0 )); then
        echo "[JD CI] ERROR: make wheel 后未找到 dist/*.whl" >&2
        exit 1
    fi

    WHEEL_FILE="${BUILT_WHEELS[$((${#BUILT_WHEELS[@]} - 1))]}"
    echo "[JD CI] hpc-ops 构建产物: ${WHEEL_FILE}"

    if [[ "${FORCE_REBUILD}" == "1" ]]; then
        rm -f "${CACHE_DIR}"/*.whl
    fi

    cp -f "${WHEEL_FILE}" "${CACHE_DIR}/"
    WHEEL_FILE="${CACHE_DIR}/$(basename "${WHEEL_FILE}")"

    echo "[JD CI] wheel 已保存到缓存: ${WHEEL_FILE}"
fi

# 复制到容器内部临时目录再安装
INSTALL_WHEEL="${INSTALL_DIR}/$(basename "${WHEEL_FILE}")"
cp -f "${WHEEL_FILE}" "${INSTALL_WHEEL}"

echo "[JD CI] 安装 hpc-ops wheel: ${INSTALL_WHEEL}"

python3 -m pip install \
    --force-reinstall \
    --no-deps \
    "${INSTALL_WHEEL}"

echo "[JD CI] hpc-ops 安装完成"

python3 -m pip show hpc-ops 2>/dev/null ||
    python3 -m pip show hpc_ops 2>/dev/null ||
    true
