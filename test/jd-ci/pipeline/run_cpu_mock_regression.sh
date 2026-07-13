#!/bin/bash
set -uo pipefail

SOURCE_PATH="${1:?Usage: run_cpu_mock_regression.sh SOURCE_PATH BASE_REF LOG_DIR}"
BASE_REF="${2:?Usage: run_cpu_mock_regression.sh SOURCE_PATH BASE_REF LOG_DIR}"
LOG_ROOT="${3:?Usage: run_cpu_mock_regression.sh SOURCE_PATH BASE_REF LOG_DIR}"

REGRESSION_DIR="${LOG_ROOT}/cpu_mock"
INVENTORY_JSON="${REGRESSION_DIR}/inventory.json"
CASES_TSV="${REGRESSION_DIR}/cases.tsv"
REPORT_JSON="${REGRESSION_DIR}/report.json"
START_TIME=$(date +%s)
FINALIZED=0
REGRESSION_EXIT_CODE=0
REGRESSION_STATUS="passed"

mkdir -p "${REGRESSION_DIR}"
printf "name\tstatus\texit_code\tlog_file\tdetail\tassertion\tduration_seconds\ttimeout_seconds\n" > "${CASES_TSV}"

export CUDA_VISIBLE_DEVICES=""
export NVIDIA_VISIBLE_DEVICES="void"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${SOURCE_PATH}/test/jd-ci:${SOURCE_PATH}/python:${PYTHONPATH:-}"

record_case() {
    local name="$1"
    local status="$2"
    local exit_code="$3"
    local log_file="$4"
    local detail="$5"
    local assertion="$6"
    local duration_seconds="$7"
    local timeout_seconds="$8"
    detail=${detail//$'\t'/ }
    detail=${detail//$'\n'/ }
    assertion=${assertion//$'\t'/ }
    assertion=${assertion//$'\n'/ }
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${name}" "${status}" "${exit_code}" "${log_file}" "${detail}" \
        "${assertion}" "${duration_seconds}" "${timeout_seconds}" \
        >> "${CASES_TSV}"
}

run_case() {
    local name="$1"
    local timeout_seconds="$2"
    local assertion="$3"
    local command_json="$4"
    local index="$5"
    local total="$6"
    local log_file="${REGRESSION_DIR}/${name}.log"
    local case_started
    local duration_seconds
    case_started=$(date +%s)
    set +e
    python3 "${SOURCE_PATH}/test/jd-ci/pipeline/case_progress.py" \
        --area "CPU and Mock Regression" \
        --case-id "${name}" \
        --index "${index}" \
        --total "${total}" \
        --assertion "${assertion}" \
        --timeout-seconds "${timeout_seconds}" \
        --kill-after-seconds 15 \
        --log-file "${log_file}" \
        --command-json "${command_json}"
    local exit_code=$?
    duration_seconds=$(( $(date +%s) - case_started ))
    if [[ ${exit_code} -eq 0 ]]; then
        record_case "${name}" passed 0 "${log_file}" "" \
            "${assertion}" "${duration_seconds}" "${timeout_seconds}"
    else
        REGRESSION_EXIT_CODE=${exit_code}
        REGRESSION_STATUS="failed"
        record_case "${name}" failed "${exit_code}" "${log_file}" \
            "fixed JD case exited with code ${exit_code}" \
            "${assertion}" "${duration_seconds}" "${timeout_seconds}"
    fi
    return ${exit_code}
}

finalize_report() {
    local original_status=$?
    if [[ ${FINALIZED} -eq 1 ]]; then
        return ${original_status}
    fi
    FINALIZED=1
    if [[ ${REGRESSION_EXIT_CODE} -eq 0 && ${original_status} -ne 0 ]]; then
        REGRESSION_EXIT_CODE=${original_status}
        REGRESSION_STATUS="failed"
    fi
    local end_time
    end_time=$(date +%s)
    python3 - "${CASES_TSV}" "${REPORT_JSON}" "${INVENTORY_JSON}" \
        "${REGRESSION_STATUS}" "${REGRESSION_EXIT_CODE}" "$((end_time - START_TIME))" \
        "${BASE_REF}" <<'PY'
import csv
import json
import os
import sys

from report.regression_report import write_regression_report

cases_path, report_path, inventory_path, status, exit_code, duration, base_ref = sys.argv[1:]
with open(cases_path, newline="", encoding="utf-8") as file:
    cases = []
    for row in csv.DictReader(file, delimiter="\t"):
        row["exit_code"] = int(row["exit_code"])
        row["duration_seconds"] = float(row["duration_seconds"])
        row["timeout_seconds"] = float(row["timeout_seconds"])
        cases.append(row)

inventory = []
if os.path.exists(inventory_path):
    with open(inventory_path, encoding="utf-8") as file:
        inventory = json.load(file)

write_regression_report(
    report_path,
    test_area="cpu_mock",
    status=status,
    cases=cases,
    duration_seconds=float(duration),
    metadata={
        "gpu_required": False,
        "gpu_hidden": os.environ.get("CUDA_VISIBLE_DEVICES") == "",
        "base_ref": base_ref,
        "selection_mode": "fixed-cumulative-jd-inventory",
        "inventory_file": inventory_path,
        "inventory_case_ids": [case["case_id"] for case in inventory],
        "exit_code": int(exit_code),
        "skip_allowed": status == "skipped",
    },
)
PY
    return ${original_status}
}

trap finalize_report EXIT
trap 'REGRESSION_EXIT_CODE=130; REGRESSION_STATUS=failed; exit 130' INT
trap 'REGRESSION_EXIT_CODE=143; REGRESSION_STATUS=failed; exit 143' TERM

cd "${SOURCE_PATH}"

if ! git rev-parse --verify "${BASE_REF}^{commit}" >/dev/null 2>&1; then
    REGRESSION_EXIT_CODE=2
    REGRESSION_STATUS="failed"
    record_case jd-test-asset-path-contract failed 2 "" \
        "cannot resolve JD CI base ref: ${BASE_REF}" \
        "JD test assets remain under test/jd-ci" 0 0
    exit ${REGRESSION_EXIT_CODE}
fi

if ! added_test_assets=$(git diff --diff-filter=A --name-only "${BASE_REF}...HEAD" -- test); then
    REGRESSION_EXIT_CODE=2
    REGRESSION_STATUS="failed"
    record_case jd-test-asset-path-contract failed 2 "" \
        "cannot audit test assets against ${BASE_REF}" \
        "JD test assets remain under test/jd-ci" 0 0
    exit ${REGRESSION_EXIT_CODE}
fi
outside_jd_ci=$(printf '%s\n' "${added_test_assets}" | awk 'NF && $0 !~ /^test\/jd-ci\//')
if [[ -n "${outside_jd_ci}" ]]; then
    REGRESSION_EXIT_CODE=2
    REGRESSION_STATUS="failed"
    echo "JD test assets must stay under test/jd-ci/:"
    printf '%s\n' "${outside_jd_ci}"
    record_case jd-test-asset-path-contract failed 2 "" \
        "new test assets outside test/jd-ci/: ${outside_jd_ci}" \
        "JD test assets remain under test/jd-ci" 0 0
    exit ${REGRESSION_EXIT_CODE}
fi

manifest_args=(
    test/jd-ci/jd_test_manifest.py
    --source "${SOURCE_PATH}"
    --category cpu
    --output "${INVENTORY_JSON}"
)
if [[ "${JD_CI_CPU_MOCK_DRY_RUN:-0}" == "1" ]]; then
    manifest_args+=(--skip-path-check)
fi
if ! python3 "${manifest_args[@]}"; then
    REGRESSION_EXIT_CODE=2
    REGRESSION_STATUS="failed"
    record_case jd-manifest-contract failed 2 "${INVENTORY_JSON}" \
        "fixed cumulative JD manifest validation failed" \
        "fixed cumulative JD manifest resolves" 0 0
    exit ${REGRESSION_EXIT_CODE}
fi

SHELL_CASE_ID="jd-ci-shell-contract"
SHELL_ASSERTION="JD-owned shell scripts pass bash syntax validation"
SHELL_TIMEOUT_SECONDS=60
MANIFEST_CASE_COUNT=$(python3 - "${INVENTORY_JSON}" <<'PY'
import json
import sys

print(len(json.load(open(sys.argv[1], encoding="utf-8"))))
PY
)
TOTAL_CASES=$((MANIFEST_CASE_COUNT + 1))
echo "[JD CI][CPU and Mock Regression][INVENTORY 1/${TOTAL_CASES}] id=${SHELL_CASE_ID} assertion=${SHELL_ASSERTION} timeout=${SHELL_TIMEOUT_SECONDS}s"
inventory_index=2
while IFS=$'\t' read -r case_id assertion timeout_seconds; do
    echo "[JD CI][CPU and Mock Regression][INVENTORY ${inventory_index}/${TOTAL_CASES}] id=${case_id} assertion=${assertion} timeout=${timeout_seconds}s"
    ((inventory_index += 1))
done < <(python3 - "${INVENTORY_JSON}" <<'PY'
import json
import sys

for case in json.load(open(sys.argv[1], encoding="utf-8")):
    print(case["case_id"], case["assertion"], case["timeout_seconds"], sep="\t")
PY
)

if [[ "${JD_CI_CPU_MOCK_DRY_RUN:-0}" == "1" ]]; then
    echo "[JD CI][CPU and Mock Regression][CASE 1/${TOTAL_CASES}][SKIP] id=${SHELL_CASE_ID} assertion=${SHELL_ASSERTION} timeout=${SHELL_TIMEOUT_SECONDS}s detail=JD_CI_CPU_MOCK_DRY_RUN=1"
    record_case "${SHELL_CASE_ID}" skipped 0 "" \
        "JD_CI_CPU_MOCK_DRY_RUN=1" "${SHELL_ASSERTION}" 0 "${SHELL_TIMEOUT_SECONDS}"
    case_index=2
    while IFS=$'\t' read -r case_id assertion timeout_seconds; do
        echo "[JD CI][CPU and Mock Regression][CASE ${case_index}/${TOTAL_CASES}][SKIP] id=${case_id} assertion=${assertion} timeout=${timeout_seconds}s detail=JD_CI_CPU_MOCK_DRY_RUN=1"
        record_case "${case_id}" skipped 0 "" \
            "JD_CI_CPU_MOCK_DRY_RUN=1" "${assertion}" 0 "${timeout_seconds}"
        ((case_index += 1))
    done < <(python3 - "${INVENTORY_JSON}" <<'PY'
import json
import sys

for case in json.load(open(sys.argv[1], encoding="utf-8")):
    print(case["case_id"], case["assertion"], case["timeout_seconds"], sep="\t")
PY
)
    REGRESSION_STATUS="skipped"
    exit 0
fi

static_command=$(python3 - <<'PY'
import json
print(json.dumps([
    "bash", "-c",
    "set -e; find test/jd-ci deploy/infer -type f -name '*.sh' -print0 | xargs -0 bash -n",
]))
PY
)
run_case "${SHELL_CASE_ID}" "${SHELL_TIMEOUT_SECONDS}" \
    "${SHELL_ASSERTION}" "${static_command}" 1 "${TOTAL_CASES}" || true

case_index=2
while IFS=$'\t' read -r case_id timeout_seconds assertion command_json; do
    run_case "${case_id}" "${timeout_seconds}" "${assertion}" \
        "${command_json}" "${case_index}" "${TOTAL_CASES}" || true
    ((case_index += 1))
done < <(python3 - "${INVENTORY_JSON}" <<'PY'
import json
import sys

for case in json.load(open(sys.argv[1], encoding="utf-8")):
    print(
        case["case_id"],
        case["timeout_seconds"],
        case["assertion"],
        json.dumps(case["command"]),
        sep="\t",
    )
PY
)

exit ${REGRESSION_EXIT_CODE}
