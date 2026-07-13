# JD CI Single-Model Regression Logging Implementation Plan

> **Completion record:** The implementation was executed task-by-task; completed steps use checkbox (`- [x]`) syntax.

**Goal:** Run every fixed JD regression with observable five-second case progress while one Qwen2.5-VL dummy-weight Server covers all model-independent HTTP and multimodal behavior.

**Architecture:** A reusable Python progress runner owns command execution, timeout, signal forwarding, stdout duplication, and the fixed five-second heartbeat. The three shell runners consume their full inventories, print every case before execution, continue after ordinary failures, and write enriched reports. The Server/API command starts Qwen2.5-VL once and executes all HTTP subcases in that process; model-specific parsers use deterministic CPU fixtures.

**Tech Stack:** Bash, Python 3, `unittest`, `subprocess`, `threading`, `requests`, Pillow, existing JD progress and JSON report utilities.

## Global Constraints

- All new files must live under `test/jd-ci/`.
- Do not execute or register SGLang native `test/registered`, `test/manual`, or `test/run_suite.py` cases.
- Every `-r` run executes the complete fixed JD inventory; current diff never selects cases.
- `-m` executes no tests; `-t` executes the full inventory unless `JD_CI_SKIP_TEST=1`.
- Start exactly one Server using `/mnt/nas/models/Qwen2.5-VL-7B-Instruct/`, `--load-format dummy`, `--disable-cuda-graph`, and `--tp-size 1`.
- Do not import upstream mock-model helpers or enable upstream `token_oracle`/KV canary machinery; the Server/API inventory covers JD behavior only.
- Heartbeat interval is always five seconds and has no environment-variable override.
- Preserve Mooncake, SGL-Kernel, container, cache, cleanup, and image publication structure.
- Never use the English or Chinese legacy phase terminology in tracked `test/jd-ci` files.

---

### Task 1: Reusable five-second case progress runner

**Files:**
- Create: `test/jd-ci/pipeline/case_progress.py`
- Create: `test/jd-ci/unit/ci/test_case_progress.py`

**Interfaces:**
- Produces: `ProgressReporter(area, case_id, index, total, assertion, timeout_seconds, action="CASE", heartbeat_seconds=5.0, stream=None)`, where `index` and `total` may be `None` for Server lifecycle actions.
- Produces: `ProgressReporter.start(phase)`, `set_phase(phase)`, and `finish(status, exit_code=0, detail="", log_file="")`.
- Produces: `run_command(command, *, reporter, log_file, timeout_seconds, kill_after_seconds) -> int`.
- Produces CLI arguments `--area`, `--case-id`, `--index`, `--total`, `--assertion`, `--timeout-seconds`, `--kill-after-seconds`, `--log-file`, and `--command-json`.

- [x] **Step 1: Write failing progress-format and heartbeat tests**

Add tests that instantiate the reporter with `heartbeat_seconds=0.01`, capture a `StringIO`, and require stable lines:

```python
reporter = ProgressReporter(
    area="CPU and Mock Regression",
    case_id="jd-example",
    index=1,
    total=2,
    assertion="example assertion",
    timeout_seconds=30,
    heartbeat_seconds=0.01,
    stream=output,
)
reporter.start("command")
time.sleep(0.03)
reporter.finish("PASS")
self.assertIn("[CASE 1/2][START] id=jd-example", output.getvalue())
self.assertIn("[RUNNING] id=jd-example phase=command", output.getvalue())
self.assertIn("elapsed=", output.getvalue())
self.assertIn("timeout=30s", output.getvalue())
self.assertIn("[PASS] id=jd-example duration=", output.getvalue())
```

Add command tests for exit `0`, exit `7`, and timeout `124`; assert the child output is present in both captured stdout and the independent case log.

- [x] **Step 2: Run the tests and verify RED**

Run:

```bash
python3 -m unittest test/jd-ci/unit/ci/test_case_progress.py -v
```

Expected: import failure because `case_progress.py` does not exist.

- [x] **Step 3: Implement progress and command execution**

Implement a lock-protected reporter whose daemon heartbeat thread waits exactly five seconds in production. `finish()` must stop and join the thread before printing the final line. Implement command execution with `subprocess.Popen(..., start_new_session=True)`, a reader thread that copies combined stdout/stderr to stdout and `log_file`, deadline enforcement, process-group `SIGTERM`, kill grace, and process-group `SIGKILL` fallback. Timeout returns `124`; forwarded interrupt and termination return `130` and `143`.

The stable line builder must emit:

```python
counter = "" if index is None else f" {index}/{total}"
prefix = f"[JD CI][{area}][{action}{counter}]"
```

and include `assertion`, `phase`, `elapsed`, `timeout`, `duration`, `exit_code`, `detail`, and `log_file` only where applicable.

- [x] **Step 4: Run the focused tests and verify GREEN**

Run the Task 1 command again. Expected: all progress tests pass in under two seconds.

- [x] **Step 5: Commit Task 1**

```bash
git add test/jd-ci/pipeline/case_progress.py test/jd-ci/unit/ci/test_case_progress.py
git commit -m "ci: add JD case progress heartbeat"
```

---

### Task 2: Make all three runners observable and non-fail-fast

**Files:**
- Modify: `test/jd-ci/pipeline/run_cpu_mock_regression.sh`
- Modify: `test/jd-ci/pipeline/run_server_api_regression.sh`
- Modify: `test/jd-ci/pipeline/run_operator_regression.sh`
- Modify: `test/jd-ci/unit/ci/test_cpu_mock_regression_runner.py`
- Modify: `test/jd-ci/unit/ci/test_server_api_regression_config.py`
- Modify: `test/jd-ci/unit/ci/test_operator_registry.py`
- Modify: `test/jd-ci/unit/ci/test_regression_report.py`

**Interfaces:**
- Consumes: Task 1 `case_progress.py` CLI.
- Produces: report cases containing `assertion`, `duration_seconds`, `timeout_seconds`, `exit_code`, `detail`, and `log_file`.
- Produces: one inventory line plus `START/RUNNING/PASS/FAIL/BLOCKED/SKIP` lines for every fixed case.

- [x] **Step 1: Write failing runner-contract tests**

Require every runner to invoke `pipeline/case_progress.py`, pass `--index`, `--total`, `--assertion`, and `--timeout-seconds`, and print the inventory before execution. Require CPU/Mock and Server/API failure blocks to continue rather than exit immediately. Extend the report test with:

```python
self.assertEqual(case["assertion"], "expected behavior")
self.assertEqual(case["timeout_seconds"], 60)
self.assertGreaterEqual(case["duration_seconds"], 0)
```

- [x] **Step 2: Run focused tests and verify RED**

```bash
python3 -m unittest \
  test/jd-ci/unit/ci/test_cpu_mock_regression_runner.py \
  test/jd-ci/unit/ci/test_server_api_regression_config.py \
  test/jd-ci/unit/ci/test_operator_registry.py \
  test/jd-ci/unit/ci/test_regression_report.py -v
```

Expected: failures for missing progress CLI fields and CPU/Server fail-fast behavior.

- [x] **Step 3: Integrate the progress CLI**

Change each TSV header to:

```text
name status exit_code log_file detail assertion duration_seconds timeout_seconds
```

Read `assertion` from manifest/spec JSON. Calculate a stable total before the loop and increment a one-based index. Before execution print every selected case as `[INVENTORY i/N]`. Replace the direct `timeout python3 ... | tee` invocation with the Task 1 CLI and measure duration for TSV output.

CPU/Mock includes the shell contract in its total. CPU/Mock and Server/API record failure, preserve a nonzero aggregate exit code, and continue. Operator preserves its existing continue behavior. Resource shortages record `BLOCKED` with the same index and assertion.

- [x] **Step 4: Verify shell syntax, focused tests, and dry-runs**

```bash
bash -n test/jd-ci/pipeline/*.sh
python3 -m unittest \
  test/jd-ci/unit/ci/test_cpu_mock_regression_runner.py \
  test/jd-ci/unit/ci/test_server_api_regression_config.py \
  test/jd-ci/unit/ci/test_operator_registry.py \
  test/jd-ci/unit/ci/test_regression_report.py -v
JD_CI_CPU_MOCK_DRY_RUN=1 \
  bash test/jd-ci/pipeline/run_cpu_mock_regression.sh "$PWD" HEAD /tmp/jd-ci-cpu-progress
JD_CI_SERVER_API_DRY_RUN=1 \
  bash test/jd-ci/pipeline/run_server_api_regression.sh "$PWD" /tmp/jd-ci-server-progress
JD_CI_OPERATOR_DRY_RUN=1 JD_CI_OPERATOR_AVAILABLE_GPUS=8 \
  bash test/jd-ci/pipeline/run_operator_regression.sh "$PWD" /tmp/jd-ci-operator-progress
```

Expected: syntax and unit tests pass; dry-run reports enumerate every fixed case with `status=skipped`.

- [x] **Step 5: Commit Task 2**

```bash
git add test/jd-ci/pipeline/run_*_regression.sh test/jd-ci/unit/ci/test_*regression* test/jd-ci/unit/ci/test_operator_registry.py
git commit -m "ci: show live JD regression case progress"
```

---

### Task 3: Expand the single Qwen2.5-VL dummy Server

**Files:**
- Modify: `test/jd-ci/pipeline/server_api_dummy_model.py`
- Modify: `test/jd-ci/pipeline/run_server_api_regression.sh`
- Modify: `test/jd-ci/unit/ci/test_server_api_regression_config.py`
- Modify: `test/jd-ci/jd_test_manifest.py`

**Interfaces:**
- Consumes: Task 1 `ProgressReporter` for Server startup and HTTP subcases.
- Produces: `ServerCase(case_id, assertion, timeout_seconds, handler_name)` and fixed `SERVER_CASES`.
- Produces: result JSON with `status`, `cases`, `server`, `server_return_code`, and `log_tail`.
- Produces: `--list-cases --output PATH`, which writes the fixed subcase inventory without starting a Server.

- [x] **Step 1: Write failing single-model inventory tests**

Require the default config and arguments:

```python
self.assertEqual(config.model_path, "/mnt/nas/models/Qwen2.5-VL-7B-Instruct/")
self.assertIn("--load-format", config.server_args)
self.assertIn("dummy", config.server_args)
self.assertIn("--tp-size", config.server_args)
self.assertIn("1", config.server_args)
```

Require exactly one call site for `popen_launch_server` and these stable subcase IDs:

```text
jd-models-endpoint
jd-text-non-streaming
jd-text-streaming
jd-image-non-streaming
jd-image-streaming
jd-broken-base64-non-streaming
jd-invalid-image-url-non-streaming
jd-broken-base64-streaming
jd-invalid-image-url-streaming
jd-ignore-eos-token-limit
jd-invalid-thinking-list
jd-invalid-thinking-dict
jd-invalid-thinking-string
jd-invalid-thinking-int
jd-tool-choice-none
```

Require the server runner to call `--list-cases` before dry-run or execution and import individual subcases from result JSON into its final report.

- [x] **Step 2: Run Server/API tests and verify RED**

```bash
python3 -m unittest test/jd-ci/unit/ci/test_server_api_regression_config.py -v
```

Expected: old Qwen3 path, missing TP1, and missing subcases.

- [x] **Step 3: Implement the fixed HTTP inventory and one Server lifecycle**

Set the default model path and use the minimal JD-owned Server arguments
`--load-format dummy --disable-cuda-graph --tp-size 1 --mem-fraction-static 0.20`.
Add the mutually exclusive CLI operation `--list-cases`, which serializes `SERVER_CASES` and exits without importing CUDA or launching a process. Pass only the JD environment overrides:

```python
{
    "JD_ENABLE_IGNORE_EOS": "true",
    "JD_DEFAULT_MAX_TOKENS": "8",
}
```

Use an inline valid PNG data URL for successful image requests and `http://127.0.0.1:1/unreachable.png` for the invalid URL. Text and image cases cover stream and non-stream responses. Invalid images assert HTTP `400` for non-streaming and any non-`500` client error for streaming. The ignore-EOS case omits explicit max tokens, sends `ignore_eos=True`, and requires `usage.completion_tokens == 8` plus `finish_reason == "length"`. Each invalid-thinking payload is its own HTTP case and requires a non-`500` response with the Server alive.

Wrap startup in a Server `ProgressReporter`; wrap each subcase in a case reporter. Continue after request failures while `process.poll() is None`. If the process exits, mark remaining cases blocked. Always write result JSON in `finally`; return nonzero when any subcase failed or blocked.

- [x] **Step 4: Parse subcases into the Server/API report**

After the grouped command exits, read its result JSON and append every result object to `cases.tsv` with its assertion, timeout, duration, status, detail, and log path. Preserve the grouped command exit code as the area exit code but do not replace individual subcase results with one opaque case.

- [x] **Step 5: Run unit tests and GPU-free dry-run**

Run the Task 3 focused test and the Server/API dry-run from Task 2. Expected: 15 fixed Server subcases appear, no Server starts during dry-run, and the report is skipped without failures.

- [x] **Step 6: Commit Task 3**

```bash
git add test/jd-ci/pipeline/server_api_dummy_model.py test/jd-ci/pipeline/run_server_api_regression.sh test/jd-ci/unit/ci/test_server_api_regression_config.py test/jd-ci/jd_test_manifest.py
git commit -m "test: cover JD APIs with one VLM server"
```

---

### Task 4: Cover DeepSeek, GLM, Kimi, and DSV4 without more Servers

**Files:**
- Modify: `test/jd-ci/unit/server/test_openai_and_function_call.py`
- Modify: `test/jd-ci/jd_test_manifest.py`
- Modify: `test/jd-ci/unit/ci/test_jd_test_manifest.py`

**Interfaces:**
- Consumes: `ChatCompletionRequest`, `OpenAIServingChat._get_reasoning_from_request`, `ReasoningParser("glm45")`, `UsageProcessor`, `KimiK2Detector`, and `encode_arguments_to_dsml`.
- Produces: deterministic JD-only CPU assertions for every model-specific repair.

- [x] **Step 1: Write failing protocol and parser tests**

First add a static contract in `test_jd_test_manifest.py` that requires four explicitly named invalid-thinking methods plus DeepSeek and GLM method names in the JD-owned test file. It must fail against the current file. Then expand invalid thinking to four visible methods for list, dict without `type`, unknown string, and integer. Create an `OpenAIServingChat` instance with `object.__new__`, set `reasoning_parser="deepseek-v4"`, and assert enabled, disabled, and absent thinking return `True`, `False`, and `False`.

Add GLM tests:

```python
parser = ReasoningParser("glm45")
reasoning, content = parser.parse_non_stream(
    "<think>reasoning<tool_call>tool payload"
)
self.assertEqual(reasoning, "reasoning")
self.assertEqual(content, "<tool_call>tool payload")
usage = UsageProcessor.calculate_response_usage([
    {"meta_info": {"prompt_tokens": 3, "completion_tokens": 5, "reasoning_tokens": 2}}
])
self.assertEqual(usage.reasoning_tokens, 2)
```

Add the equivalent incremental GLM parser assertion. Retain and run the existing Kimi quoted-object and DSV4 malformed-arguments assertions.

- [x] **Step 2: Run the JD OpenAI test and verify RED**

```bash
PYTHONPATH=python:test/jd-ci python3 test/jd-ci/unit/server/test_openai_and_function_call.py -v
```

Expected: the static manifest contract fails because the required invalid variants, DeepSeek-specific switch tests, and GLM fixture tests are absent.

- [x] **Step 3: Implement the deterministic tests**

Use `CustomTestCase`; do not launch a Server. Keep each invalid input as an individual method so CI logs and unittest output name the precise failing form. Map the existing DSV4/OpenAI JD commit to this expanded case in the manifest without adding upstream test commands.

- [x] **Step 4: Run focused and manifest tests**

```bash
PYTHONPATH=python:test/jd-ci python3 test/jd-ci/unit/server/test_openai_and_function_call.py -v
python3 -m unittest test/jd-ci/unit/ci/test_jd_test_manifest.py -v
```

Expected: all tests pass; `missing_commits`, `unexpected_commits`, and upstream command violations remain empty.

- [x] **Step 5: Commit Task 4**

```bash
git add test/jd-ci/unit/server/test_openai_and_function_call.py test/jd-ci/jd_test_manifest.py test/jd-ci/unit/ci/test_jd_test_manifest.py
git commit -m "test: cover JD model-specific protocol fixes"
```

---

### Task 5: Documentation and complete verification

**Files:**
- Modify: `test/jd-ci/README.md`
- Modify: `test/jd-ci/single_model_regression_logging_implementation_plan.md`

**Interfaces:**
- Consumes: all earlier task outputs.
- Produces: user-facing coverage and log examples matching the implementation.

- [x] **Step 1: Update README**

Document the one-Server boundary, exact Qwen2.5-VL dummy path, 15 Server/API subcases, model-specific CPU fixtures, five-second log protocol, failure continuation behavior, and the statement that this does not prove model accuracy or other checkpoint loading.

- [x] **Step 2: Run complete local verification**

```bash
bash -n test/jd-ci/run_jd_ci.sh test/jd-ci/env/*.sh test/jd-ci/pipeline/*.sh
python3 -m unittest discover -s test/jd-ci/unit/ci -p 'test_*.py' -v
PYTHONPATH=python:test/jd-ci python3 test/jd-ci/unit/server/test_openai_and_function_call.py -v
JD_CI_CPU_MOCK_DRY_RUN=1 bash test/jd-ci/pipeline/run_cpu_mock_regression.sh "$PWD" HEAD /tmp/jd-ci-cpu-final
JD_CI_SERVER_API_DRY_RUN=1 bash test/jd-ci/pipeline/run_server_api_regression.sh "$PWD" /tmp/jd-ci-server-final
JD_CI_OPERATOR_DRY_RUN=1 JD_CI_OPERATOR_AVAILABLE_GPUS=8 bash test/jd-ci/pipeline/run_operator_regression.sh "$PWD" /tmp/jd-ci-operator-final
rg -ni 'sta''ge|阶''段|TO''DO|TB''D|待''定' test/jd-ci
git diff --check
```

Expected: syntax and tests pass; dry-runs contain complete inventories; terminology scan has no matches; diff check is clean.

- [x] **Step 3: Mark this plan complete and commit**

Change every checkbox in this plan to `[x]`, then:

```bash
git add test/jd-ci/README.md test/jd-ci/single_model_regression_logging_implementation_plan.md
git commit -m "docs: explain JD live regression progress"
```

- [x] **Step 4: Push the implementation branch**

```bash
git push origin codex/jd-ci-cumulative-regression
```

Expected: remote branch resolves to the same SHA as local HEAD.
