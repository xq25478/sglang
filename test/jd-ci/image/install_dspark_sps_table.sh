#!/usr/bin/env bash
set -euo pipefail

SOURCE_TABLE="${1:?usage: install_dspark_sps_table.sh SOURCE_TABLE [TARGET_TABLE]}"
# Default target mirrors the source basename so any backend variant works.
TARGET_TABLE="${2:-/$(basename "${SOURCE_TABLE}")}"

sha256_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

# Derive the expected hash from the source file itself so any backend
# variant (humming, flashinfer-cutlass, ...) is supported without editing
# this script.
EXPECTED_SHA256="$(sha256_file "${SOURCE_TABLE}")"

install -m 0644 "${SOURCE_TABLE}" "${TARGET_TABLE}"

TARGET_SHA256="$(sha256_file "${TARGET_TABLE}")"
if [[ "${TARGET_SHA256}" != "${EXPECTED_SHA256}" ]]; then
    echo "[JD CI] ERROR: installed DSpark SPS table SHA256 mismatch: ${TARGET_SHA256}" >&2
    exit 1
fi

echo "[JD CI] DSpark SPS table installed: ${TARGET_TABLE} (${TARGET_SHA256})"
