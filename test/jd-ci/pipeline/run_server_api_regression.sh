#!/bin/bash
set -uo pipefail

SOURCE_PATH="${1:?Usage: run_server_api_regression.sh SOURCE_PATH LOG_DIR}"
LOG_ROOT="${2:?Usage: run_server_api_regression.sh SOURCE_PATH LOG_DIR}"

REGRESSION_DIR="${LOG_ROOT}/server_api"
MANIFEST_JSON="${REGRESSION_DIR}/manifest.json"
INVENTORY_JSON="${REGRESSION_DIR}/inventory.json"
CASES_TSV="${REGRESSION_DIR}/cases.tsv"
REPORT_JSON="${REGRESSION_DIR}/report.json"
START_TIME=$(date +%s)
REGRESSION_EXIT_CODE=0
REGRESSION_STATUS="passed"
FINALIZED=0

mkdir -p "${REGRESSION_DIR}"
printf "name\tstatus\texit_code\tlog_file\tdetail\tassertion\tduration_seconds\ttimeout_seconds\n" > "${CASES_TSV}"
export CUDA_VISIBLE_DEVICES="${JD_CI_SERVER_API_GPU_ID:-0}"
export NVIDIA_VISIBLE_DEVICES="${JD_CI_SERVER_API_GPU_ID:-0}"
export PYTHONPATH="${SOURCE_PATH}/test/jd-ci:${SOURCE_PATH}/test/jd-ci/pipeline:${SOURCE_PATH}/python:${PYTHONPATH:-}"

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
    python3 - "${CASES_TSV}" "${REPORT_JSON}" "${INVENTORY_JSON}" \
        "${REGRESSION_STATUS}" "${REGRESSION_EXIT_CODE}" "$((end_time - START_TIME))" <<'PY'
import csv
import json
import os
import sys

from report.regression_report import write_regression_report

cases_path, report_path, inventory_path, status, exit_code, duration = sys.argv[1:]
with open(cases_path, newline="", encoding="utf-8") as file:
    cases = []
    for row in csv.DictReader(file, delimiter="\t"):
        row["exit_code"] = int(row["exit_code"])
        row["duration_seconds"] = float(row["duration_seconds"])
        row["timeout_seconds"] = float(row["timeout_seconds"])
        cases.append(row)
inventory = []
if os.path.exists(inventory_path):
    inventory = json.load(open(inventory_path, encoding="utf-8"))

write_regression_report(
    report_path,
    test_area="server_api",
    status=status,
    cases=cases,
    duration_seconds=float(duration),
    metadata={
        "gpu_required": True,
        "required_gpus": 1,
        "selection_mode": "fixed-cumulative-jd-server-cases",
        "server_case_ids": [case["case_id"] for case in inventory],
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

if ! python3 test/jd-ci/jd_test_manifest.py \
    --source "${SOURCE_PATH}" --category server --output "${MANIFEST_JSON}"; then
    REGRESSION_EXIT_CODE=2
    REGRESSION_STATUS="failed"
    record_case jd-manifest-contract failed 2 "${MANIFEST_JSON}" \
        "fixed cumulative JD manifest validation failed" \
        "fixed cumulative JD server manifest resolves" 0 0
    exit ${REGRESSION_EXIT_CODE}
fi
if ! python3 test/jd-ci/pipeline/server_api_dummy_model.py \
    --list-cases --output "${INVENTORY_JSON}"; then
    REGRESSION_EXIT_CODE=2
    REGRESSION_STATUS="failed"
    record_case jd-server-inventory failed 2 "${INVENTORY_JSON}" \
        "failed to resolve fixed Server and API subcases" \
        "all fixed Server and API subcases resolve without CUDA" 0 0
    exit ${REGRESSION_EXIT_CODE}
fi

TOTAL_CASES=$(python3 - "${INVENTORY_JSON}" <<'PY'
import json
import sys

print(len(json.load(open(sys.argv[1], encoding="utf-8"))))
PY
)
inventory_index=1
while IFS=$'\t' read -r case_id assertion timeout_seconds; do
    echo "[JD CI][Server and API Regression][INVENTORY ${inventory_index}/${TOTAL_CASES}] id=${case_id} assertion=${assertion} timeout=${timeout_seconds}s"
    ((inventory_index += 1))
done < <(python3 - "${INVENTORY_JSON}" <<'PY'
import json
import sys

for case in json.load(open(sys.argv[1], encoding="utf-8")):
    print(case["case_id"], case["assertion"], case["timeout_seconds"], sep="\t")
PY
)

if [[ "${JD_CI_SERVER_API_DRY_RUN:-0}" == "1" ]]; then
    case_index=1
    while IFS=$'\t' read -r case_id assertion timeout_seconds; do
        echo "[JD CI][Server and API Regression][CASE ${case_index}/${TOTAL_CASES}][SKIP] id=${case_id} assertion=${assertion} timeout=${timeout_seconds}s detail=JD_CI_SERVER_API_DRY_RUN=1"
        record_case "${case_id}" skipped 0 "" \
            "JD_CI_SERVER_API_DRY_RUN=1" "${assertion}" 0 "${timeout_seconds}"
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

gpu_count=$(python3 - <<'PY'
import torch
print(torch.cuda.device_count())
PY
) || {
    REGRESSION_EXIT_CODE=2
    REGRESSION_STATUS="failed"
    record_case jd-gpu-detection failed 2 "" \
        "failed to query visible CUDA devices" \
        "exactly one GPU is visible to the Server regression" 0 0
    exit ${REGRESSION_EXIT_CODE}
}
if [[ "${gpu_count}" != "1" ]]; then
    REGRESSION_EXIT_CODE=2
    REGRESSION_STATUS="failed"
    record_case jd-gpu-detection failed 2 "" \
        "Server and API Regression requires exactly one visible GPU, got ${gpu_count}" \
        "exactly one GPU is visible to the Server regression" 0 0
    exit ${REGRESSION_EXIT_CODE}
fi

result_json="${REGRESSION_DIR}/jd-server-api-regressions.json"
log_file="${REGRESSION_DIR}/jd-server-api-regressions.log"
read -r aggregate_case_id aggregate_timeout_seconds aggregate_assertion command_json \
    < <(python3 - "${MANIFEST_JSON}" <<'PY'
import json
import sys

case = json.load(open(sys.argv[1], encoding="utf-8"))[0]
print(
    case["case_id"],
    case["timeout_seconds"],
    case["assertion"].replace(" ", "__JD_SPACE__"),
    json.dumps(case["command"], separators=(",", ":")),
)
PY
)
aggregate_assertion=${aggregate_assertion//__JD_SPACE__/ }
resolved_command_json=$(python3 - "${command_json}" "${result_json}" <<'PY'
import json
import sys

command = [sys.argv[2] if part == "{result}" else part for part in json.loads(sys.argv[1])]
print(json.dumps(command))
PY
)
set +e
python3 "${SOURCE_PATH}/test/jd-ci/pipeline/case_progress.py" \
    --area "Server and API Regression Orchestration" \
    --case-id "${aggregate_case_id}" \
    --index 1 \
    --total 1 \
    --assertion "${aggregate_assertion}" \
    --timeout-seconds "${aggregate_timeout_seconds}" \
    --kill-after-seconds 30 \
    --log-file "${log_file}" \
    --command-json "${resolved_command_json}"
case_exit_code=$?
if [[ ${case_exit_code} -ne 0 ]]; then
    REGRESSION_EXIT_CODE=${case_exit_code}
    REGRESSION_STATUS="failed"
fi

python3 - "${INVENTORY_JSON}" "${result_json}" "${CASES_TSV}" \
    "${log_file}" "${case_exit_code}" <<'PY'
import csv
import json
import sys
from pathlib import Path

inventory_path, result_path, cases_path, log_file, aggregate_exit_code = sys.argv[1:]
inventory = json.loads(Path(inventory_path).read_text(encoding="utf-8"))
if Path(result_path).is_file():
    result = json.loads(Path(result_path).read_text(encoding="utf-8"))
else:
    result = {}
returned_cases = {
    case.get("case_id", case.get("name")): case
    for case in result.get("cases", [])
}
aggregate_exit_code = int(aggregate_exit_code)
with open(cases_path, "a", newline="", encoding="utf-8") as file:
    writer = csv.writer(file, delimiter="\t", lineterminator="\n")
    for index, expected in enumerate(inventory, start=1):
        case_id = expected["case_id"]
        case = returned_cases.get(case_id)
        if case is None:
            detail = (
                "aggregate Server command ended before this case produced a result"
            )
            case = {
                "status": "blocked",
                "exit_code": aggregate_exit_code or 3,
                "detail": detail,
                "duration_seconds": 0.0,
            }
            print(
                f"[JD CI][Server and API Regression][CASE {index}/{len(inventory)}]"
                f"[BLOCKED] id={case_id} duration=0.0s "
                f"exit_code={case['exit_code']} detail={detail}",
                flush=True,
            )
        detail = " ".join(str(case.get("detail", "")).replace("\t", " ").splitlines())
        assertion = " ".join(
            str(case.get("assertion", expected["assertion"]))
            .replace("\t", " ")
            .splitlines()
        )
        writer.writerow(
            [
                case_id,
                case.get("status", "failed"),
                int(case.get("exit_code", aggregate_exit_code or 1)),
                case.get("log_file") or log_file,
                detail,
                assertion,
                float(case.get("duration_seconds", 0.0)),
                float(case.get("timeout_seconds", expected["timeout_seconds"])),
            ]
        )
PY

exit ${REGRESSION_EXIT_CODE}
