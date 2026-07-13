#!/bin/bash
set -u

CPU_COUNT="${1:-$(nproc)}"

is_positive_integer() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

if ! is_positive_integer "${CPU_COUNT}"; then
    echo "[SGLang CI] ERROR: CPU count must be a positive integer, got '${CPU_COUNT}'" >&2
    exit 2
fi

# Keep parallelism primarily at Ninja's independent translation-unit level.
# One Ninja job per host CPU maximizes translation-unit concurrency without
# multiplying every job by a second nvcc thread pool.
DEFAULT_MAX_JOBS=${CPU_COUNT}

BUILD_MAX_JOBS="${JD_CI_SGL_KERNEL_BUILD_MAX_JOBS:-${DEFAULT_MAX_JOBS}}"
NVCC_THREADS="${JD_CI_SGL_KERNEL_NVCC_THREADS:-1}"

if ! is_positive_integer "${BUILD_MAX_JOBS}"; then
    echo "[SGLang CI] ERROR: JD_CI_SGL_KERNEL_BUILD_MAX_JOBS must be a positive integer, got '${BUILD_MAX_JOBS}'" >&2
    exit 2
fi
if ! is_positive_integer "${NVCC_THREADS}"; then
    echo "[SGLang CI] ERROR: JD_CI_SGL_KERNEL_NVCC_THREADS must be a positive integer, got '${NVCC_THREADS}'" >&2
    exit 2
fi

printf '%s %s\n' "${BUILD_MAX_JOBS}" "${NVCC_THREADS}"
