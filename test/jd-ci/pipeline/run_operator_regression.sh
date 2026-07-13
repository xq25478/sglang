#!/bin/bash
set -uo pipefail

SOURCE_PATH="${1:?Usage: run_operator_regression.sh SOURCE_PATH LOG_DIR}"
LOG_ROOT="${2:?Usage: run_operator_regression.sh SOURCE_PATH LOG_DIR}"

REGRESSION_DIR="${LOG_ROOT}/operator"
SPECS_JSON="${REGRESSION_DIR}/specs.json"
CASES_TSV="${REGRESSION_DIR}/cases.tsv"
REPORT_JSON="${REGRESSION_DIR}/report.json"
START_TIME=$(date +%s)
REGRESSION_EXIT_CODE=0
REGRESSION_STATUS="passed"
AVAILABLE_GPUS=0
FINALIZED=0

mkdir -p "${REGRESSION_DIR}"
printf "name\tstatus\texit_code\tlog_file\tdetail\tassertion\tduration_seconds\ttimeout_seconds\n" > "${CASES_TSV}"
export PYTHONPATH="${SOURCE_PATH}/test/jd-ci:${SOURCE_PATH}/test/jd-ci/operators:${SOURCE_PATH}/python:${PYTHONPATH:-}"

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
    python3 - "${CASES_TSV}" "${REPORT_JSON}" "${SPECS_JSON}" \
        "${REGRESSION_STATUS}" "${REGRESSION_EXIT_CODE}" "$((end_time - START_TIME))" \
        "${AVAILABLE_GPUS}" <<'PY'
import csv
import json
import os
import sys

from report.regression_report import write_regression_report

cases_path, report_path, specs_path, status, exit_code, duration, available_gpus = sys.argv[1:]
with open(cases_path, newline="", encoding="utf-8") as file:
    cases = []
    for row in csv.DictReader(file, delimiter="\t"):
        row["exit_code"] = int(row["exit_code"])
        row["duration_seconds"] = float(row["duration_seconds"])
        row["timeout_seconds"] = float(row["timeout_seconds"])
        cases.append(row)
specs = []
if os.path.exists(specs_path):
    specs = json.load(open(specs_path, encoding="utf-8"))

write_regression_report(
    report_path,
    test_area="operator",
    status=status,
    cases=cases,
    duration_seconds=float(duration),
    metadata={
        "gpu_required": True,
        "required_gpus": max((int(spec["min_gpus"]) for spec in specs), default=0),
        "available_gpus": int(available_gpus),
        "selection_mode": "fixed-cumulative-jd-operators",
        "operator_case_ids": [spec["name"] for spec in specs],
        "operator_pairs": sorted({spec["operator"] for spec in specs}),
        "skip_allowed": status == "skipped",
        "exit_code": int(exit_code),
    },
)
PY
    return ${original_status}
}

trap finalize_report EXIT
trap 'REGRESSION_EXIT_CODE=130; REGRESSION_STATUS=failed; exit 130' INT
trap 'REGRESSION_EXIT_CODE=143; REGRESSION_STATUS=failed; exit 143' TERM

cd "${SOURCE_PATH}"

if ! python3 test/jd-ci/operator_registry.py --output "${SPECS_JSON}"; then
    REGRESSION_EXIT_CODE=2
    REGRESSION_STATUS="failed"
    record_case jd-operator-registry failed 2 "${SPECS_JSON}" \
        "failed to resolve the fixed JD operator inventory" \
        "fixed JD operator inventory resolves" 0 0
    exit ${REGRESSION_EXIT_CODE}
fi

TOTAL_CASES=$(python3 - "${SPECS_JSON}" <<'PY'
import json
import sys

print(len(json.load(open(sys.argv[1], encoding="utf-8"))))
PY
)
inventory_index=1
while IFS=$'\t' read -r name role min_gpus timeout_seconds assertion; do
    echo "[JD CI][Operator Correctness and Performance Regression][INVENTORY ${inventory_index}/${TOTAL_CASES}] id=${name} role=${role} min_gpus=${min_gpus} assertion=${assertion} timeout=${timeout_seconds}s"
    ((inventory_index += 1))
done < <(python3 - "${SPECS_JSON}" <<'PY'
import json
import sys

for spec in json.load(open(sys.argv[1], encoding="utf-8")):
    print(
        spec["name"],
        spec["role"],
        spec["min_gpus"],
        spec["timeout_seconds"],
        spec["assertion"],
        sep="\t",
    )
PY
)

if [[ -n "${JD_CI_OPERATOR_AVAILABLE_GPUS:-}" ]]; then
    AVAILABLE_GPUS="${JD_CI_OPERATOR_AVAILABLE_GPUS}"
else
    AVAILABLE_GPUS=$(python3 - <<'PY'
import torch
print(torch.cuda.device_count())
PY
) || {
        REGRESSION_EXIT_CODE=2
        REGRESSION_STATUS="failed"
        record_case jd-gpu-detection failed 2 "" \
            "failed to query visible CUDA devices" \
            "visible CUDA device count can be determined" 0 0
        exit ${REGRESSION_EXIT_CODE}
    }
fi

case_index=1
while IFS=$'\t' read -r name role min_gpus timeout_seconds assertion command_json; do
    log_file="${REGRESSION_DIR}/${name}.log"
    if (( AVAILABLE_GPUS < min_gpus )); then
        REGRESSION_EXIT_CODE=3
        REGRESSION_STATUS="blocked"
        echo "[JD CI][Operator Correctness and Performance Regression][CASE ${case_index}/${TOTAL_CASES}][START] id=${name} assertion=${assertion} timeout=${timeout_seconds}s"
        echo "[JD CI][Operator Correctness and Performance Regression][CASE ${case_index}/${TOTAL_CASES}][BLOCKED] id=${name} duration=0.0s exit_code=3 detail=requires ${min_gpus} GPUs, available ${AVAILABLE_GPUS}"
        record_case "${name}" blocked 3 "${log_file}" \
            "requires ${min_gpus} GPUs, available ${AVAILABLE_GPUS}" \
            "${assertion}" 0 "${timeout_seconds}"
        ((case_index += 1))
        continue
    fi

    if [[ "${JD_CI_OPERATOR_DRY_RUN:-0}" == "1" ]]; then
        echo "[JD CI][Operator Correctness and Performance Regression][CASE ${case_index}/${TOTAL_CASES}][SKIP] id=${name} assertion=${assertion} timeout=${timeout_seconds}s detail=JD_CI_OPERATOR_DRY_RUN=1"
        record_case "${name}" skipped 0 "${log_file}" \
            "fixed ${role} case; JD_CI_OPERATOR_DRY_RUN=1" \
            "${assertion}" 0 "${timeout_seconds}"
        ((case_index += 1))
        continue
    fi

    case_started=$(date +%s)
    set +e
    python3 "${SOURCE_PATH}/test/jd-ci/pipeline/case_progress.py" \
        --area "Operator Correctness and Performance Regression" \
        --case-id "${name}" \
        --index "${case_index}" \
        --total "${TOTAL_CASES}" \
        --assertion "${assertion}" \
        --timeout-seconds "${timeout_seconds}" \
        --kill-after-seconds 30 \
        --log-file "${log_file}" \
        --command-json "${command_json}"
    case_exit_code=$?
    duration_seconds=$(( $(date +%s) - case_started ))
    if [[ ${case_exit_code} -ne 0 ]]; then
        REGRESSION_EXIT_CODE=${case_exit_code}
        REGRESSION_STATUS="failed"
        record_case "${name}" failed "${case_exit_code}" "${log_file}" \
            "fixed JD ${role} case exited with code ${case_exit_code}" \
            "${assertion}" "${duration_seconds}" "${timeout_seconds}"
        ((case_index += 1))
        continue
    else
        record_case "${name}" passed 0 "${log_file}" "${role}" \
            "${assertion}" "${duration_seconds}" "${timeout_seconds}"
    fi
    ((case_index += 1))
done < <(python3 - "${SPECS_JSON}" <<'PY'
import json
import sys

for spec in json.load(open(sys.argv[1], encoding="utf-8")):
    print(
        spec["name"],
        spec["role"],
        spec["min_gpus"],
        spec["timeout_seconds"],
        spec["assertion"],
        json.dumps(spec["command"]),
        sep="\t",
    )
PY
)

if [[ "${JD_CI_OPERATOR_DRY_RUN:-0}" == "1" && ${REGRESSION_EXIT_CODE} -eq 0 ]]; then
    REGRESSION_STATUS="skipped"
fi
exit ${REGRESSION_EXIT_CODE}
