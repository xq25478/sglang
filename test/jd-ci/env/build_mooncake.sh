#!/bin/bash
# Mooncake 一体化构建脚本：拉取代码 → 安装依赖 → 编译 → 打包 wheel → 验证
# 用法: build_mooncake.sh <MCAKE_UPPER_PATH> [MCAKE_OPTION] [MCAKE_CLEAR]
#   MCAKE_UPPER_PATH : Mooncake 代码的上级目录。
#   MCAKE_OPTION     : 编译模式，"te"(默认，含 CUDA/EP/NVLink) 或 "store"(纯 store，无 CUDA/EP)。
#   MCAKE_CLEAR      : 编译后是否清理源码，1=清理(默认) 0=保留。
# Mooncake 分支会根据当前环境已安装的 mooncake-transfer-engine 版本自动推导为 JD-v<version>。
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Error: illegal number of arguments $#"
    echo "Usage: build_mooncake.sh <MCAKE_UPPER_PATH> [MCAKE_OPTION] [MCAKE_CLEAR]"
    exit 1
fi

MCAKE_VERSION=$(pip list 2>/dev/null | grep -i '^mooncake-transfer-engine' | awk '{print $2}' | head -1)
MCAKE_BRANCH="JD-v${MCAKE_VERSION}"
MCAKE_UPPER_PATH="$1"
MCAKE_OPTION="${2:-te}"
MCAKE_CLEAR="${3:-1}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MCAKE_REPO_URL="git@coding.jd.com:llm-project/Mooncake.git"
MCAKE_REPO_DIR="Mooncake"
MCAKE_PATH="$MCAKE_UPPER_PATH/$MCAKE_REPO_DIR"

OS_RELEASE_FILE=${OS_RELEASE_FILE:-/etc/os-release}

MOONCAKE_BUILD_JOBS="${MOONCAKE_BUILD_JOBS:-$(nproc)}"
MOONCAKE_ENABLE_ETCD="${MOONCAKE_ENABLE_ETCD:-OFF}"
MOONCAKE_ENABLE_P2P_STORE="${MOONCAKE_ENABLE_P2P_STORE:-OFF}"
MOONCAKE_ENABLE_STORE_GO="${MOONCAKE_ENABLE_STORE_GO:-OFF}"
MOONCAKE_EP_TORCH_VERSIONS="${MOONCAKE_EP_TORCH_VERSIONS:-}"
MOONCAKE_TORCH_CUDA_ARCH_LIST="${MOONCAKE_TORCH_CUDA_ARCH_LIST:-8.9;9.0;10.0;10.3;12.0}"
MOONCAKE_ENGINE_CACHE_DIR="${MOONCAKE_ENGINE_CACHE_DIR:-}"
MOONCAKE_WHEEL_CACHE_DIR="${MOONCAKE_WHEEL_CACHE_DIR:-}"
MOONCAKE_FORCE_REBUILD="${MOONCAKE_FORCE_REBUILD:-1}"
MOONCAKE_REQUIRE_CACHE="${MOONCAKE_REQUIRE_CACHE:-0}"

_PKG_SOURCES_READY=0
_CMAKE_ARGS_CONFIGURED=0
PYTHON_VERSION=""
PYTHON_BIN=""

# ──────────────────────────────────────────────────────────────────────────────
# 颜色 & 通用工具函数
# ──────────────────────────────────────────────────────────────────────────────

GREEN="\033[0;32m"
BLUE="\033[0;34m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
NC="\033[0m"

print_section() {
    echo -e "\n${BLUE}=== $1 ===${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ ERROR: $1${NC}"
    exit 1
}

check_success() {
    if [ $? -ne 0 ]; then
        print_error "$1"
    fi
}

version_ge() {
    local current=$1
    local minimum=$2
    [[ "$(printf '%s\n%s\n' "${minimum}" "${current}" | sort -V | head -n1)" == "${minimum}" ]]
}

require_cmd() {
    local cmd=$1
    if ! command -v "${cmd}" >/dev/null 2>&1; then
        print_error "missing required command: ${cmd}"
    fi
}

append_env_flag() {
    local var_name=$1
    local flag=$2
    local current="${!var_name:-}"
    if [[ " ${current} " == *" ${flag} "* ]]; then
        return
    fi
    if [[ -n "${current}" ]]; then
        export "${var_name}=${current} ${flag}"
    else
        export "${var_name}=${flag}"
    fi
}

read_os_release_value() {
    local key="$1"
    awk -F= -v key="$key" '
        $1 == key {
            value = $0
            sub(/^[^=]*=/, "", value)
            gsub(/^"|"$/, "", value)
            print value
            exit
        }
    ' "$OS_RELEASE_FILE"
}

cmake_arg_value() {
    local key=$1
    local arg
    local value=""
    for arg in "${CMAKE_ARGS[@]}"; do
        case "${arg}" in
            -D${key}=*)
                value="${arg#-D${key}=}"
                ;;
        esac
    done
    printf '%s\n' "${value}"
}

cmake_arg_enabled() {
    local value
    value="$(cmake_arg_value "$1")"
    case "${value}" in
        ON|on|On|TRUE|true|True|YES|yes|Yes|1)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

cmake_requires_go() {
    cmake_arg_enabled USE_ETCD \
        || cmake_arg_enabled STORE_USE_ETCD \
        || cmake_arg_enabled STORE_USE_K8S_LEASE \
        || cmake_arg_enabled WITH_P2P_STORE \
        || cmake_arg_enabled WITH_STORE_GO
}

configure_cmake_args() {
    if [[ "${_CMAKE_ARGS_CONFIGURED}" -eq 1 ]]; then
        return
    fi

    if [ "$MCAKE_OPTION" != "store" ]; then
        DEFAULT_CMAKE_ARGS=(
            -DUSE_CUDA=ON
            -DUSE_CXL=ON
            -DWITH_STORE=ON
            -DWITH_P2P_STORE=${MOONCAKE_ENABLE_P2P_STORE}
            -DWITH_STORE_RUST=ON
            -DWITH_STORE_GO=${MOONCAKE_ENABLE_STORE_GO}
            -DUSE_TCP=ON
            -DUSE_HTTP=ON
            -DUSE_ETCD=${MOONCAKE_ENABLE_ETCD}
            -DSTORE_USE_ETCD=${MOONCAKE_ENABLE_ETCD}
            -DBUILD_UNIT_TESTS=OFF
            -DBUILD_EXAMPLES=ON
            -DBUILD_BENCHMARK=ON
            -DUSE_MNNVL=ON
            -DWITH_TE=ON
            -DWITH_METRICS=ON
            -DWITH_EP=ON
            -DEP_TORCH_VERSIONS=${MOONCAKE_EP_TORCH_VERSIONS}
            -DTORCH_CUDA_ARCH_LIST=${MOONCAKE_TORCH_CUDA_ARCH_LIST}
            -DCMAKE_BUILD_TYPE=Release
            -DUSE_INTRA_NVLINK=ON
        )
        print_section "Mooncake compile $MCAKE_OPTION != store"
    else
        DEFAULT_CMAKE_ARGS=(
            -DUSE_CUDA=OFF
            -DUSE_CXL=ON
            -DWITH_STORE=ON
            -DWITH_P2P_STORE=${MOONCAKE_ENABLE_P2P_STORE}
            -DWITH_STORE_RUST=ON
            -DWITH_STORE_GO=${MOONCAKE_ENABLE_STORE_GO}
            -DUSE_TCP=ON
            -DUSE_HTTP=ON
            -DUSE_ETCD=${MOONCAKE_ENABLE_ETCD}
            -DSTORE_USE_ETCD=${MOONCAKE_ENABLE_ETCD}
            -DBUILD_UNIT_TESTS=OFF
            -DBUILD_EXAMPLES=ON
            -DBUILD_BENCHMARK=ON
            -DUSE_MNNVL=OFF
            -DWITH_TE=ON
            -DWITH_METRICS=ON
            -DWITH_EP=OFF
            -DEP_TORCH_VERSIONS=${MOONCAKE_EP_TORCH_VERSIONS}
            -DTORCH_CUDA_ARCH_LIST=${MOONCAKE_TORCH_CUDA_ARCH_LIST}
            -DCMAKE_BUILD_TYPE=Release
            -DUSE_INTRA_NVLINK=OFF
        )
        print_section "Mooncake compile $MCAKE_OPTION == store"
    fi

    if [[ -n "${MOONCAKE_CMAKE_ARGS:-}" ]]; then
        read -r -a CMAKE_ARGS <<< "${MOONCAKE_CMAKE_ARGS}"
        echo "input mooncake args:${CMAKE_ARGS[*]}"
    else
        CMAKE_ARGS=("${DEFAULT_CMAKE_ARGS[@]}")
        echo "default mooncake args:${CMAKE_ARGS[*]}"
    fi

    if [[ -n "${MOONCAKE_EXTRA_CMAKE_ARGS:-}" ]]; then
        read -r -a EXTRA_CMAKE_ARGS <<< "${MOONCAKE_EXTRA_CMAKE_ARGS}"
        CMAKE_ARGS+=("${EXTRA_CMAKE_ARGS[@]}")
    fi

    _CMAKE_ARGS_CONFIGURED=1
}

ensure_python_context() {
    if [[ -n "${PYTHON_VERSION}" ]]; then
        return
    fi

    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    PYTHON_BIN="python${PYTHON_VERSION}"
    require_cmd "${PYTHON_BIN}"
}

mooncake_wheel_cache_path() {
    printf '%s\n' "${MOONCAKE_WHEEL_CACHE_DIR}"
}

find_cached_mooncake_wheel() {
    local cache_path="$1"

    if [[ ! -d "${cache_path}" ]]; then
        return 1
    fi

    find "${cache_path}" -maxdepth 1 -type f -name 'mooncake_transfer_engine*.whl' | sort | tail -1
}

install_cached_mooncake_wheel_if_available() {
    if [[ "${MOONCAKE_FORCE_REBUILD}" == "1" ]]; then
        if [[ "${MOONCAKE_REQUIRE_CACHE}" == "1" ]]; then
            print_error "MOONCAKE_FORCE_REBUILD=1 conflicts with MOONCAKE_REQUIRE_CACHE=1"
        fi
        echo "[Mooncake CI] MOONCAKE_FORCE_REBUILD=1，跳过 wheel cache 命中检查"
        return 1
    fi

    if [[ -z "${MOONCAKE_WHEEL_CACHE_DIR}" ]]; then
        if [[ "${MOONCAKE_REQUIRE_CACHE}" == "1" ]]; then
            print_error "MOONCAKE_REQUIRE_CACHE=1 requires MOONCAKE_WHEEL_CACHE_DIR"
        fi
        echo "[Mooncake CI] MOONCAKE_WHEEL_CACHE_DIR 未设置，跳过 wheel cache"
        return 1
    fi

    local cache_path
    local whl_file
    ensure_python_context
    cache_path=$(mooncake_wheel_cache_path)

    echo "[Mooncake CI] PYTHON_VERSION=${PYTHON_VERSION}"
    echo "[Mooncake CI] Mooncake wheel cache path: ${cache_path}"
    whl_file=$(find_cached_mooncake_wheel "${cache_path}" || true)
    if [[ -z "${whl_file}" ]]; then
        if [[ "${MOONCAKE_REQUIRE_CACHE}" == "1" ]]; then
            print_error "Mooncake wheel cache miss: ${cache_path}"
        fi
        echo "[Mooncake CI] Mooncake wheel cache miss，继续源码编译"
        return 1
    fi

    print_section "Installing Mooncake wheel from cache"
    echo "[Mooncake CI] Cached wheel: ${whl_file}"
    verify_update_wheel_version "${cache_path}"

    if [ "$MCAKE_OPTION" != "store" ]; then
        print_section "Verifying cached cuda wheel nvlink"
        verify_cuda_wheel_contents
        verify_intra_node_nvlink
        save_mooncake_engine_cache "${whl_file}"
        print_success "Verifying cached cuda wheel nvlink successfully"
    fi

    print_success "Mooncake wheel cache hit installed successfully"
    return 0
}

save_mooncake_wheel_cache() {
    local wheel_dir="$1"
    local whl_file
    local cache_path
    local lock_file

    if [[ -z "${MOONCAKE_WHEEL_CACHE_DIR}" ]]; then
        return
    fi

    whl_file=$(find "${wheel_dir}" -maxdepth 1 -type f -name 'mooncake_transfer_engine*.whl' | sort | tail -1)
    if [[ -z "${whl_file}" ]]; then
        print_error "No mooncake_transfer_engine wheel found for cache in ${wheel_dir}"
    fi

    cache_path=$(mooncake_wheel_cache_path)
    lock_file="${cache_path}.lock"

    mkdir -p "${cache_path}"
    (
        flock 8
        rm -f "${cache_path}"/mooncake_transfer_engine*.whl
        cp -f "${whl_file}" "${cache_path}/"
    ) 8>"${lock_file}"

    print_success "Mooncake wheel cached: ${cache_path}/$(basename "${whl_file}")"
}

save_mooncake_engine_cache() {
    local cached_wheel="${1:-}"
    local engine_so

    if [[ "${MCAKE_OPTION}" == "store" || -z "${MOONCAKE_ENGINE_CACHE_DIR}" ]]; then
        return
    fi

    engine_so=$(get_installed_engine_so)
    if [[ ! -f "${engine_so}" ]]; then
        print_error "installed Mooncake engine.so not found: ${engine_so}"
    fi

    if [[ -z "${cached_wheel}" && -n "${MOONCAKE_WHEEL_CACHE_DIR}" ]]; then
        local cache_path
        cache_path=$(mooncake_wheel_cache_path)
        cached_wheel=$(find_cached_mooncake_wheel "${cache_path}" || true)
    fi

    mkdir -p "${MOONCAKE_ENGINE_CACHE_DIR}"
    cp -f "${engine_so}" "${MOONCAKE_ENGINE_CACHE_DIR}/engine.so"
    chmod +x "${MOONCAKE_ENGINE_CACHE_DIR}/engine.so"
    cat > "${MOONCAKE_ENGINE_CACHE_DIR}/build_info.txt" <<EOF
MCAKE_BRANCH=${MCAKE_BRANCH}
MCAKE_OPTION=${MCAKE_OPTION}
PYTHON_VERSION=${PYTHON_VERSION}
CMAKE_ARGS=${CMAKE_ARGS[*]}
WHEEL=${cached_wheel}
BUILT_AT=$(date -Iseconds)
EOF

    print_success "Mooncake engine.so cached: ${MOONCAKE_ENGINE_CACHE_DIR}/engine.so"
}

# ──────────────────────────────────────────────────────────────────────────────
# 拉取代码
# ──────────────────────────────────────────────────────────────────────────────

ensure_git_installed() {
    if command -v git >/dev/null 2>&1; then
        echo "[Mooncake CI] git 已存在: $(git --version)"
        return
    fi

    echo "[Mooncake CI] git 未安装，开始安装 git 和 openssh-client..."
    detect_os

    if [ "$OS" = "ubuntu" ]; then
        ubuntu_container_apt_sources
        apt-get install -y --no-install-recommends git openssh-client
    elif [ "$OS" = "debian" ]; then
        debain_container_apt_sources
        apt-get install -y --no-install-recommends git openssh-client
    elif [ "$OS" = "centos" ] || [ "$OS" = "rhel" ] || [ "$OS" = "rocky" ] || [ "$OS" = "almalinux" ] || [ "$OS" = "euleros" ] || [ "$OS" = "openeuler" ]; then
        if [ "${_PKG_SOURCES_READY}" -eq 0 ]; then
            yum clean all
            yum makecache
            _PKG_SOURCES_READY=1
        fi
        yum install -y git openssh-clients
    else
        print_error "Unsupported OS: $OS, cannot install git"
    fi

    require_cmd git
    print_success "git 安装成功: $(git --version)"
}

clear_history() {
    if [ -n "$MCAKE_UPPER_PATH" ] && [ -d "$MCAKE_PATH" ]; then
        rm -rf "$MCAKE_PATH"
    fi
    if [ -n "$MCAKE_PATH" ] && [ -d "$MCAKE_PATH/extern" ]; then
        rm -rf "${MCAKE_PATH}/extern/"*
    fi
}

fix_submodule_urls() {
    local gitmodules="${1}/.gitmodules"
    local pybind11_url="git@coding.jd.com:llm-project/pybind11.git"
    local yalantinglibs_url="git@coding.jd.com:llm-project/yalantinglibs.git"

    if [ ! -f "$gitmodules" ]; then
        print_error ".gitmodules file not found at $gitmodules"
    fi

    if ! grep -q '\[submodule "extern/pybind11"\]' "$gitmodules"; then
        print_error "submodule extern/pybind11 not found in .gitmodules"
    fi

    if ! grep -q '\[submodule "extern/yalantinglibs"\]' "$gitmodules"; then
        print_error "submodule extern/yalantinglibs not found in .gitmodules"
    fi

    git -C "$1" config -f .gitmodules submodule.extern/pybind11.url "$pybind11_url"
    git -C "$1" config -f .gitmodules submodule.extern/yalantinglibs.url "$yalantinglibs_url"

    print_success "submodule URLs replaced successfully"
}

pull_code() {
    if [ ! -d "$MCAKE_UPPER_PATH" ]; then
        print_error "clone Mooncake Path:$MCAKE_UPPER_PATH not exist"
    fi

    ensure_git_installed

    print_section "Pulling Mooncake source code"

    cd "$MCAKE_UPPER_PATH"
    check_success "Failed to change to $MCAKE_UPPER_PATH directory"

    clear_history

    echo "git clone Mooncake $MCAKE_REPO_URL branch=$MCAKE_BRANCH..."
    git clone -q -b "$MCAKE_BRANCH" "$MCAKE_REPO_URL"
    check_success "Failed to git clone Mooncake"

    cd "$MCAKE_PATH"
    check_success "Failed to change to $MCAKE_PATH directory"
    echo "[Mooncake CI] Mooncake branch: $(git rev-parse --abbrev-ref HEAD), commit: $(git rev-parse HEAD)"

    echo "change Mooncake .gitmodules..."
    fix_submodule_urls "$MCAKE_PATH"

    print_section "Initializing Git Submodules"

    if [ -f "${MCAKE_PATH}/.gitmodules" ]; then
        echo "Enter repository root: ${MCAKE_PATH}"
        cd "${MCAKE_PATH}/"
        check_success "Failed to change to repository root directory"

        echo "Initializing git submodules..."
        git submodule sync --recursive -q
        check_success "Failed to sync git submodules"
        git submodule update --init --recursive -q
        check_success "Failed to initialize git submodules"

        print_success "Git submodules initialized and updated successfully"
        git submodule foreach --quiet 'echo "[Mooncake CI] submodule $name: commit=$(git rev-parse HEAD), describe=$(git describe --tags --always 2>/dev/null)"'
    else
        echo -e "${YELLOW}No .gitmodules file found. Skipping...${NC}"
        exit 1
    fi

    print_section "Mooncake and submodule coding pull finish"
    echo -e "You can now start container compile Mooncake"
}

# ──────────────────────────────────────────────────────────────────────────────
# 安装依赖
# ──────────────────────────────────────────────────────────────────────────────

detect_os() {
    if [ -f "$OS_RELEASE_FILE" ]; then
        ID=$(read_os_release_value ID)
        VERSION_ID=$(read_os_release_value VERSION_ID)
        OS=$(echo "$ID" | tr '[:upper:]' '[:lower:]')
        OS_VERSION=$VERSION_ID
    elif [ -f /etc/redhat-release ]; then
        OS="centos"
    else
        print_error "Cannot detect OS. Supported OS: Ubuntu, Debian, CentOS, RHEL, Rocky, AlmaLinux, EulerOS, and openEuler."
    fi
    echo -e "${GREEN}Detected OS: $OS ${OS_VERSION:-unknown}${NC}"
}

ubuntu_container_apt_sources() {
    if [ "${_PKG_SOURCES_READY}" -eq 1 ]; then
        return
    fi

    local codename
    codename=$(. /etc/os-release && echo "${VERSION_CODENAME:-${UBUNTU_CODENAME:-}}")
    if [[ -z "${codename}" ]]; then
        print_error "failed to detect Ubuntu codename from /etc/os-release"
    fi

    local mirror="${MOONCAKE_APT_MIRROR:-http://mirrors.jdcloudcs.com/ubuntu}"
    local source_file="/etc/apt/sources.list.d/mooncake-ci-${codename}.list"
    local pin_file="/etc/apt/preferences.d/mooncake-ci-${codename}"

    cat > "${source_file}" <<EOFAPT
deb ${mirror}/ ${codename} main restricted universe multiverse
deb ${mirror}/ ${codename}-security main restricted universe multiverse
deb ${mirror}/ ${codename}-updates main restricted universe multiverse
EOFAPT

    cat > "${pin_file}" <<EOFPIN
Package: *
Pin: release n=${codename}
Pin-Priority: 1001
EOFPIN

    echo "[Mooncake CI] 已配置容器匹配的 apt 源: ${source_file}"
    echo "[Mooncake CI] Ubuntu codename=${codename}, mirror=${mirror}"
    apt-get update
    _PKG_SOURCES_READY=1
}

debain_container_apt_sources() {
    if [ "${_PKG_SOURCES_READY}" -eq 1 ]; then
        return
    fi

    local codename
    codename=$(. /etc/os-release && echo "${VERSION_CODENAME:-${DEBAIN_CODENAME:-}}")
    if [[ -z "${codename}" ]]; then
        print_error "failed to detect Debian codename from /etc/os-release"
    fi

    local mirror="${MOONCAKE_APT_MIRROR:-http://mirrors.jdcloudcs.com/debian}"
    local source_file="/etc/apt/sources.list.d/mooncake-ci-${codename}.list"
    local pin_file="/etc/apt/preferences.d/mooncake-ci-${codename}"

    > /etc/apt/sources.list 2>/dev/null || true
    rm -f /etc/apt/sources.list.d/*.sources /etc/apt/sources.list.d/*.list 2>/dev/null || true

    cat > "${source_file}" <<EOFAPT
deb ${mirror}/ ${codename} main contrib non-free non-free-firmware
deb ${mirror}-security/ ${codename}-security main contrib non-free non-free-firmware
deb ${mirror}/ ${codename}-updates main contrib non-free non-free-firmware
EOFAPT

    cat > "${pin_file}" <<EOFPIN
Package: *
Pin: release n=${codename}
Pin-Priority: 1001

Package: *
Pin: release n=${codename}-security
Pin-Priority: 1001

Package: *
Pin: release n=${codename}-updates
Pin-Priority: 1001
EOFPIN

    echo "[Mooncake CI] 已配置容器匹配的 apt 源: ${source_file}"
    echo "[Mooncake CI] Debian codename=${codename}, mirror=${mirror}"
    apt-get update
    _PKG_SOURCES_READY=1
}

install_go_if_needed() {
    export PATH="/usr/local/go/bin:${PATH}"
    if command -v go >/dev/null 2>&1; then
        echo "[Mooncake CI] Go 已存在: $(go version)"
        return
    fi
    echo "[Mooncake CI] apt 安装 golang-go..."
    apt-get install -y --no-install-recommends golang-go
    require_cmd go
    echo "[Mooncake CI] Go 安装成功: $(go version)"
}

find_yaml_cpp_config_dir() {
    local config
    config=$(find /usr /usr/local -type f \( \
        -name 'yaml-cppConfig.cmake' -o \
        -name 'yaml-cpp-config.cmake' \
    \) -print -quit 2>/dev/null || true)
    if [[ -n "${config}" ]]; then
        dirname "${config}"
    fi
}

ensure_yaml_cpp_dev() {
    local yaml_cpp_dir
    yaml_cpp_dir=$(find_yaml_cpp_config_dir)
    if [[ -z "${yaml_cpp_dir}" ]]; then
        echo "[Mooncake CI] 未找到 yaml-cpp CMake config，补装 libyaml-cpp-dev..."
        if [ "$OS" = "ubuntu" ] || [ "$OS" = "debian" ]; then
            apt-get install -y --no-install-recommends libyaml-cpp-dev
        elif [ "$OS" = "centos" ] || [ "$OS" = "rhel" ] || [ "$OS" = "rocky" ] || [ "$OS" = "almalinux" ] || [ "$OS" = "euleros" ] || [ "$OS" = "openeuler" ]; then
            yum install -y yaml-cpp-devel
        fi
        yaml_cpp_dir=$(find_yaml_cpp_config_dir)
    fi

    if [[ -z "${yaml_cpp_dir}" ]]; then
        print_error "libyaml-cpp-dev installed but yaml-cpp CMake config still not found"
    fi

    CMAKE_ARGS+=("-Dyaml-cpp_DIR=${yaml_cpp_dir}")
    echo "[Mooncake CI] yaml-cpp_DIR=${yaml_cpp_dir}"
}

install_dependencies() {
    if [ ! -d "$MCAKE_PATH" ]; then
        print_error "Mooncake Path:$MCAKE_PATH not exist"
    fi

    if [ $(id -u) -ne 0 ]; then
        print_error "Require root permission, try sudo"
    fi

    detect_os

    print_section "Updating package lists"
    if [ "$OS" = "ubuntu" ] || [ "$OS" = "debian" ]; then
        if [ "$OS" = "ubuntu" ]; then
            ubuntu_container_apt_sources
        else
            debain_container_apt_sources
        fi
    elif [ "$OS" = "centos" ] || [ "$OS" = "rhel" ] || [ "$OS" = "rocky" ] || [ "$OS" = "almalinux" ] || [ "$OS" = "euleros" ] || [ "$OS" = "openeuler" ]; then
        if [ "${_PKG_SOURCES_READY}" -eq 0 ]; then
            yum clean all
            yum makecache
            check_success "Failed to update package lists"
            _PKG_SOURCES_READY=1
        fi
    else
        print_error "Unsupported OS: $OS"
    fi

    print_section "Installing system packages"
    echo -e "${YELLOW}This may take a few minutes...${NC}"

    if [ "$OS" = "ubuntu" ] || [ "$OS" = "debian" ]; then
        SYSTEM_PACKAGES="build-essential \
                         cmake \
                         ninja-build \
                         git \
                         wget \
                         unzip \
                         libibverbs-dev \
                         libgoogle-glog-dev \
                         libgtest-dev \
                         libjsoncpp-dev \
                         libunwind-dev \
                         libnuma-dev \
                         libpython3-dev \
                         libboost-all-dev \
                         libssl-dev \
                         libgrpc-dev \
                         libgrpc++-dev \
                         libprotobuf-dev \
                         libyaml-cpp-dev \
                         protobuf-compiler-grpc \
                         libcurl4-openssl-dev \
                         libhiredis-dev \
                         liburing-dev \
                         libjemalloc-dev \
                         libmsgpack-dev \
                         libzstd-dev \
                         libasio-dev \
                         libxxhash-dev \
                         pkg-config \
                         patchelf \
                         cargo \
                         rustc \
                         libc6-dev \
                         libc-bin \
                         pybind11-dev \
                         libgflags-dev"

        if apt-cache show libmsgpack-cxx-dev >/dev/null 2>&1; then
            SYSTEM_PACKAGES="${SYSTEM_PACKAGES} libmsgpack-cxx-dev"
        fi

        apt-get install -y $SYSTEM_PACKAGES
        check_success "Failed to install system packages"

    elif [ "$OS" = "centos" ] || [ "$OS" = "rhel" ] || [ "$OS" = "rocky" ] || [ "$OS" = "almalinux" ] || [ "$OS" = "euleros" ] || [ "$OS" = "openeuler" ] || [ "$OS" = "vesselos" ]; then
        SYSTEM_PACKAGES="@development \
                         cmake \
                         git \
                         wget \
                         rdma-core-devel \
                         glog-devel \
                         gtest-devel \
                         jsoncpp-devel \
                         libunwind-devel \
                         numactl-devel \
                         python3-devel \
                         boost-devel \
                         openssl-devel \
                         grpc-devel \
                         protobuf-devel \
                         yaml-cpp-devel \
                         grpc-plugins \
                         libcurl-devel \
                         hiredis-devel \
                         liburing-devel \
                         jemalloc-devel \
                         pkgconf-pkg-config \
                         elfutils-libelf-devel \
                         patchelf \
                         xxhash-devel \
                         libbsd-devel"

        yum install -y $SYSTEM_PACKAGES
        check_success "Failed to install system packages"
    else
        print_error "Unsupported OS: $OS"
    fi

    print_success "System packages installed successfully"

    if cmake_requires_go; then
        install_go_if_needed
        export GOPROXY="${GOPROXY:-https://goproxy.cn,https://goproxy.io,direct}"
    else
        echo "[Mooncake CI] 跳过 Go 安装：USE_ETCD/STORE_USE_ETCD/WITH_P2P_STORE/WITH_STORE_GO 均未启用"
    fi

    export LIBRARY_PATH="/usr/local/lib:${LIBRARY_PATH:-}"
    export LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH:-}"

    require_cmd cmake
    if cmake_requires_go; then
        require_cmd go
    fi

    print_section "Installing yalantinglibs"
    cd "${MCAKE_PATH}/extern/yalantinglibs"
    check_success "Failed to change to yalantinglibs submodule directory"

    rm -rf build
    mkdir -p build
    check_success "Failed to create build directory"
    cd build
    check_success "Failed to change to build directory"

    echo "Configuring yalantinglibs..."
    cmake .. -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARK=OFF -DBUILD_UNIT_TESTS=OFF
    check_success "Failed to configure yalantinglibs"

    echo "Building yalantinglibs (using $(nproc) cores)..."
    cmake --build . -j$(nproc)
    check_success "Failed to build yalantinglibs"

    echo "Installing yalantinglibs..."
    cmake --install .
    check_success "Failed to install yalantinglibs"

    print_success "yalantinglibs installed successfully"
    cd "${MCAKE_PATH}"

    ensure_yaml_cpp_dev

    print_section "Verifying essential build tools"
    if [ "$OS" = "ubuntu" ] || [ "$OS" = "debian" ]; then
        if ! command -v getconf >/dev/null 2>&1; then
            print_error "getconf not found after installing system packages."
        fi
        if ! command -v ldd >/dev/null 2>&1; then
            print_error "ldd not found after installing system packages."
        fi
        print_success "getconf found: $(getconf --version 2>&1 | head -1)"
        print_success "ldd found: $(ldd --version 2>&1 | head -1)"
    fi

    print_success "All dependencies have been successfully installed"
}

# ──────────────────────────────────────────────────────────────────────────────
# 编译、打包与验证
# ──────────────────────────────────────────────────────────────────────────────

setup_mooncake_tmp_dir() {
    local tmp_dir="${MOONCAKE_TMP_DIR:-/tmp/mooncake-build}"
    mkdir -p "${tmp_dir}"
    chmod 700 "${tmp_dir}"

    export TMPDIR="${tmp_dir}"
    export TMP="${tmp_dir}"
    export TEMP="${tmp_dir}"
    append_env_flag CFLAGS -pipe
    append_env_flag CXXFLAGS -pipe

    echo "[Mooncake CI] TMPDIR=${TMPDIR}"
    echo "[Mooncake CI] CFLAGS=${CFLAGS:-}"
    echo "[Mooncake CI] CXXFLAGS=${CXXFLAGS:-}"
}

setup_cuda_build_env() {
    CUDA_ROOT="${CUDA_HOME:-/usr/local/cuda}"
    CUDA_LIB64_DIR="${CUDA_ROOT}/lib64"
    CUDA_STUBS_DIR="${CUDA_LIB64_DIR}/stubs"

    if [[ -d "${CUDA_LIB64_DIR}" ]]; then
        export LIBRARY_PATH="${CUDA_LIB64_DIR}:${LIBRARY_PATH:-}"
        export LD_LIBRARY_PATH="${CUDA_LIB64_DIR}:${LD_LIBRARY_PATH:-}"
    fi

    if [[ -d "${CUDA_STUBS_DIR}" ]]; then
        export LIBRARY_PATH="${CUDA_STUBS_DIR}:${LIBRARY_PATH:-}"
        if [[ " ${CMAKE_ARGS[*]} " != *"CMAKE_EXE_LINKER_FLAGS"* ]]; then
            CMAKE_ARGS+=("-DCMAKE_EXE_LINKER_FLAGS=-L${CUDA_STUBS_DIR}")
        fi
    fi

    export EP_TORCH_VERSIONS="${MOONCAKE_EP_TORCH_VERSIONS}"
    export TORCH_CUDA_ARCH_LIST="${MOONCAKE_TORCH_CUDA_ARCH_LIST}"
    echo "[Mooncake CI] CUDA_ROOT=${CUDA_ROOT}"
    echo "[Mooncake CI] EP_TORCH_VERSIONS=${EP_TORCH_VERSIONS}"
    echo "[Mooncake CI] TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
}

build_nvlink_allocator() {
    local allocator_dir="mooncake-transfer-engine/nvlink-allocator"
    local allocator_out="$(pwd)/build/mooncake-transfer-engine/nvlink-allocator"

    if [[ ! -d "${allocator_dir}" ]]; then
        print_error "nvlink allocator source missing: ${allocator_dir}"
    fi

    mkdir -p "${allocator_out}"
    echo "[Mooncake CI] 编译 nvlink_allocator.so: ${allocator_out}"
    (
        cd "${allocator_dir}"
        export PATH="/usr/local/nvidia/bin:/usr/local/nvidia/lib64:${PATH}"
        export LIBRARY_PATH="${CUDA_STUBS_DIR:-}:${CUDA_LIB64_DIR:-}:${LIBRARY_PATH:-}"
        export LD_LIBRARY_PATH="${CUDA_LIB64_DIR:-}:${LD_LIBRARY_PATH:-}"
        bash build.sh "${allocator_out}/"
    )

    if [[ ! -f "${allocator_out}/nvlink_allocator.so" ]]; then
        print_error "nvlink_allocator.so build output missing: ${allocator_out}/nvlink_allocator.so"
    fi
}

clean_wheel_artifacts() {
    local wheel_dir="$1"
    if [ -d "$wheel_dir" ]; then
        rm -rf "$wheel_dir/dist" "$wheel_dir/build" "$wheel_dir"/*.egg-info "$wheel_dir/repaired_wheels_"*
        echo "Cleaned wheel artifacts in $wheel_dir"
    fi
}

verify_update_wheel_version() {
    local wheel_dir="$1"
    local whl_file

    whl_file=$(find "$wheel_dir" -name "mooncake_transfer_engine*.whl" 2>/dev/null | head -1)
    if [ -z "$whl_file" ]; then
        print_error "No mooncake_transfer_engine wheel found in $wheel_dir"
    fi

    local whl_version
    whl_version=$(basename "$whl_file" | sed -n 's/mooncake_transfer_engine[^-]*-\([^-]*\)-.*/\1/p')

    local installed_version
    local installed_pkg=$(pip list 2>/dev/null | grep -i "^mooncake-transfer-engine" | awk '{print $1}' | head -1)
    if [ -n "$installed_pkg" ]; then
        installed_version=$(pip show "$installed_pkg" 2>/dev/null | grep "^Version:" | awk '{print $2}')
    fi

    if [ -z "$installed_version" ]; then
        print_error "mooncake-transfer-engine is not installed in the environment"
    fi

    if [ "$whl_version" != "$installed_version" ]; then
        print_error "Version mismatch: wheel=$whl_version, installed=$installed_version"
    fi

    print_success "Version verified: $whl_version"

    pip uninstall -y mooncake-transfer-engine
    check_success "Failed to uninstall old mooncake-transfer-engine"

    pip show mooncake-transfer-engine 2>/dev/null && print_error "Package still exists after uninstall" || echo "Confirmed: package uninstalled"

    pip install --force-reinstall --no-deps "$whl_file"
    check_success "Failed to install new mooncake-transfer-engine wheel"

    print_success "mooncake-transfer-engine upgraded to $whl_version"
}

get_installed_engine_so() {
    python3 - <<'PYENGINEPATH'
from mooncake import engine
print(engine.__file__)
PYENGINEPATH
}

detect_installed_mooncake_package() {
    python3 - <<'PYMOONCAKEPKG'
import importlib.metadata as metadata

candidates = (
    "mooncake-transfer-engine-cuda13",
    "mooncake-transfer-engine",
    "mooncake-transfer-engine-non-cuda",
    "mooncake-transfer-engine-npu",
)
for package in candidates:
    try:
        print(package, metadata.version(package))
        raise SystemExit(0)
    except metadata.PackageNotFoundError:
        pass
raise SystemExit(1)
PYMOONCAKEPKG
}

verify_intra_node_nvlink() {
    local engine_so
    engine_so=$(get_installed_engine_so)
    echo "[Mooncake CI] Mooncake engine.so: ${engine_so}"

    if command -v strings >/dev/null 2>&1; then
        if strings "${engine_so}" | grep -Fq "Protocol 'nvlink_intra' requires -DUSE_INTRA_NVLINK=ON"; then
            print_error "installed Mooncake engine.so still lacks USE_INTRA_NVLINK"
        fi
    fi

    local verify_log
    verify_log=$(mktemp "${TMPDIR:-/tmp}/mooncake-intra-nvlink.XXXXXX.log")
    set +e
    MC_INTRANODE_NVLINK=true GLOG_logtostderr=1 python3 - <<'PYVERIFY' 2>&1 | tee "${verify_log}"
from mooncake.engine import TransferEngine

engine = TransferEngine()
ret = engine.initialize("127.0.0.1:12345", "P2PHANDSHAKE", "rdma", "")
print("init ret:", ret)
raise SystemExit(0 if ret == 0 else 1)
PYVERIFY
    local verify_status=${PIPESTATUS[0]}
    set -e

    if [[ ${verify_status} -ne 0 ]]; then
        print_error "Mooncake Intra-Node NVLink 初始化探针失败"
    fi

    if ! grep -Fq "Using Intra-Node NVLink transport" "${verify_log}"; then
        echo "[Mooncake CI] ERROR: Mooncake 未使用 Intra-Node NVLink transport，验证日志如下:"
        cat "${verify_log}"
        exit 1
    fi

    rm -f "${verify_log}"
    echo "[Mooncake CI] Mooncake Intra-Node NVLink transport 验证成功"
}

verify_cuda_wheel_contents() {
    local mooncake_dir
    mooncake_dir=$(python3 - <<'PYMOONCAKEDIR'
import pathlib
import mooncake
print(pathlib.Path(mooncake.__file__).resolve().parent)
PYMOONCAKEDIR
)

    local required_files=(
        engine.so
        libasio.so
        store.so
        mooncake_master
        mooncake_client
        transfer_engine_bench
        nvlink_allocator.so
        allocator.py
        fabric_allocator_utils.py
    )

    if cmake_arg_enabled USE_ETCD || cmake_arg_enabled STORE_USE_ETCD; then
        required_files+=(libetcd_wrapper.so)
    else
        echo "[Mooncake CI] ETCD 未启用，跳过 libetcd_wrapper.so 产物校验"
    fi

    local file
    for file in "${required_files[@]}"; do
        if [[ ! -e "${mooncake_dir}/${file}" ]]; then
            print_error "Mooncake wheel 关键产物缺失: ${mooncake_dir}/${file}"
        fi
    done

    local old_nullglob=0
    if shopt -q nullglob; then
        old_nullglob=1
    fi
    shopt -s nullglob

    local pattern
    for pattern in \
        'ep_*.so' \
        'pg_*.so'; do
        local matches=("${mooncake_dir}"/${pattern})
        if [[ ${#matches[@]} -eq 0 ]]; then
            print_error "Mooncake wheel EP/PG 产物缺失: ${pattern}"
        fi
    done

    if [[ ${old_nullglob} -eq 0 ]]; then
        shopt -u nullglob
    fi

    echo "[Mooncake CI] Mooncake CUDA wheel 产物验证成功"
}

compile_and_package() {
    if [ ! -d "$MCAKE_PATH" ]; then
        print_error "Mooncake Path:$MCAKE_PATH not exist"
    fi

    configure_cmake_args

    print_section "Installing Mooncake dependencies"
    install_dependencies
    print_success "Mooncake dependencies installed successfully"

    setup_mooncake_tmp_dir
    if [ "$MCAKE_OPTION" != "store" ]; then
        setup_cuda_build_env
    fi

    ensure_python_context
    echo "[Mooncake CI] PYTHON_VERSION=${PYTHON_VERSION}"
    echo "[Mooncake CI] MOONCAKE_CMAKE_ARGS=${CMAKE_ARGS[*]}"

    print_section "Compiling Mooncake library"
    cd "$MCAKE_PATH"
    check_success "Failed to change to $MCAKE_PATH directory"

    rm -rf build
    mkdir -p build
    check_success "Failed to create build directory"

    cd build
    check_success "Failed to change to build directory"

    echo "Configuring Mooncake..."
    cmake .. "${CMAKE_ARGS[@]}" -DCMAKE_CXX_FLAGS="-w" -DCMAKE_C_FLAGS="-w"
    check_success "Failed to configure Mooncake"

    echo "Building Mooncake (using ${MOONCAKE_BUILD_JOBS} cores)..."
    make -j${MOONCAKE_BUILD_JOBS}
    check_success "Failed to build Mooncake"
    print_success "Compiling Mooncake library successfully"
    cd "$MCAKE_PATH"

    if [ "$MCAKE_OPTION" != "store" ]; then
        build_nvlink_allocator
    fi

    WHEEL_PATH="dist"

    print_section "Updating Mooncake wheel package, python version:$PYTHON_VERSION"

    echo "clear Mooncake wheel artifacts"
    clean_wheel_artifacts "$MCAKE_PATH/mooncake-wheel"

    echo "packaging Mooncake wheel"
    BUILD_DIR="build" bash "$MCAKE_PATH/scripts/build_wheel.sh" "$PYTHON_VERSION" "$WHEEL_PATH"
    check_success "Failed to generate Mooncake wheel package"

    echo "ensure Mooncake wheel version then update"
    verify_update_wheel_version "$MCAKE_PATH/mooncake-wheel/$WHEEL_PATH"
    save_mooncake_wheel_cache "$MCAKE_PATH/mooncake-wheel/$WHEEL_PATH"

    print_success "Update Mooncake wheel package successfully"

    if [ "$MCAKE_OPTION" != "store" ]; then
        print_section "Verifying image cuda wheel nvlink"
        echo "verifying image cuda wheel"
        verify_cuda_wheel_contents
        echo "verifying image nvlink"
        verify_intra_node_nvlink
        save_mooncake_engine_cache
        print_success "Verifying image cuda wheel nvlink successfully"
    fi

    if [ "$MCAKE_CLEAR" != "0" ]; then
        print_section "Delete Mooncake Compile files"
        cd "$MCAKE_UPPER_PATH"
        check_success "Failed to change to $MCAKE_UPPER_PATH directory"
        rm -rf "$MCAKE_REPO_DIR"
        print_success "Delete Mooncake Compile files succeed"
    fi
}

# ──────────────────────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────────────────────

echo "[Mooncake CI] ======================================"
echo "[Mooncake CI] MCAKE_BRANCH=${MCAKE_BRANCH}"
echo "[Mooncake CI] MCAKE_UPPER_PATH=${MCAKE_UPPER_PATH}"
echo "[Mooncake CI] MCAKE_OPTION=${MCAKE_OPTION}"
echo "[Mooncake CI] MCAKE_CLEAR=${MCAKE_CLEAR}"
echo "[Mooncake CI] MOONCAKE_ENGINE_CACHE_DIR=${MOONCAKE_ENGINE_CACHE_DIR:-<unset>}"
echo "[Mooncake CI] MOONCAKE_WHEEL_CACHE_DIR=${MOONCAKE_WHEEL_CACHE_DIR:-<unset>}"
echo "[Mooncake CI] MOONCAKE_FORCE_REBUILD=${MOONCAKE_FORCE_REBUILD}"
echo "[Mooncake CI] MOONCAKE_REQUIRE_CACHE=${MOONCAKE_REQUIRE_CACHE}"
echo "[Mooncake CI] ======================================"

configure_cmake_args
if install_cached_mooncake_wheel_if_available; then
    print_section "Build Complete"
    echo -e "${GREEN}Mooncake cache install finished successfully!${NC}"
    exit 0
fi

pull_code
compile_and_package

print_section "Build Complete"
echo -e "${GREEN}Mooncake build finished successfully!${NC}"
