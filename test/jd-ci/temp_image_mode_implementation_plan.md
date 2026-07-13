# JD CI 临时验证镜像 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不改变 Mooncake、SGL-Kernel 主体编译命令和三类固定回归内容的前提下，为 `run_jd_ci.sh` 落地严格的 `-r`、`-m`、`-t` 三种模式，并保留两个旧事件名入口。

**Architecture:** 参数解析、环境变量校验和帮助输出放在任何 Git/Docker 操作之前。正式评审与正式合入共享持久缓存，临时验证仅使用完整 commit SHA 隔离目录；主容器与 Mooncake-store 容器全部成功后，才统一决定是否发布两张镜像。

**Tech Stack:** Bash、Python `unittest` 静态与轻量行为契约、Docker CLI（只在真实流水线运行）。

## Global Constraints

- 所有新增文件只能位于 `test/jd-ci/`。
- 不改变 `build_mooncake.sh` 的 Mooncake 主体编译命令。
- 不改变 `build_sgl_kernel.sh` 的 SGL-Kernel 主体编译命令和并行度策略。
- `-r` 每次固定执行 CPU/Mock、Server/API、算子正确性与性能三类全部回归。
- `-m` 只使用 `-r` 生成的正式缓存；任一缓存 miss 必须失败。
- `-t` 默认编译组件并执行全部测试，只有用户显式设置对应的 `0/1` 变量才能跳过。
- `note__merge_request` 必须等价于 `-r`；`merge_request__merged` 必须等价于 `-m`。
- 不使用旧的流水线分段术语。
- 本次验证不得创建或推送镜像。

---

### Task 1: 参数、帮助和旧事件名契约

**Files:**
- Modify: `test/jd-ci/unit/ci/test_internal_ci_contract.py`
- Modify: `test/jd-ci/run_jd_ci.sh`

**Interfaces:**
- Consumes: 现有 `EVENT_TYPE` 下游接口。
- Produces: `CI_MODE=review|merge|temp-image`、`EVENT_TYPE`、`RUN_CI_TESTS`、`PUBLISH_IMAGES`。

- [x] **Step 1: Write the failing tests**

```python
cases = (
    ((), "review", "note__merge_request"),
    (("-r",), "review", "note__merge_request"),
    (("--review",), "review", "note__merge_request"),
    (("note__merge_request",), "review", "note__merge_request"),
    (("-m",), "merge", "merge_request__merged"),
    (("--merge",), "merge", "merge_request__merged"),
    (("merge_request__merged",), "merge", "merge_request__merged"),
    (("-t",), "temp-image", "temp_image"),
    (("--temp-image",), "temp-image", "temp_image"),
)
```

另外直接运行完整脚本的 `-h`、未知参数和非法 `0/1` 值，断言帮助状态码为 0、错误状态码为 2，并检查三个临时选项均出现在帮助中。

- [x] **Step 2: Run tests and confirm RED**

Run: `python3 test/jd-ci/unit/ci/test_internal_ci_contract.py -v`

Expected: 新增的长参数、`-t`、帮助或非法值断言失败，证明现有入口未实现新协议。

- [x] **Step 3: Implement the entry protocol**

```bash
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
        RUN_CI_TESTS=$((1 - JD_CI_SKIP_TEST))
        PUBLISH_IMAGES=1
        ;;
esac
```

帮助函数必须先于 `CI_SCRIPT_PATH`，`-h` 和参数错误必须在工作区检查与 Docker 查询之前退出。

- [x] **Step 4: Run tests and confirm GREEN**

Run: `python3 test/jd-ci/unit/ci/test_internal_ci_contract.py -v`

Expected: 参数、帮助和旧事件名契约通过。

### Task 2: 正式缓存与 commit 临时目录

**Files:**
- Modify: `test/jd-ci/unit/ci/test_internal_ci_contract.py`
- Modify: `test/jd-ci/run_jd_ci.sh`
- Modify: `test/jd-ci/env/build_sgl_kernel.sh`

**Interfaces:**
- Consumes: `CI_MODE` 和三个临时选项。
- Produces: `JD_CI_ARTIFACT_SCOPE`、两个组件 scope、三个 wheel/cache 挂载目录、Mooncake cache 策略。

- [x] **Step 1: Write the failing cache-policy tests**

```python
self.assertIn('case "${CI_MODE}" in', script)
self.assertIn('${CI_ARTIFACT_ROOT}/tmp_artifacts/${COMMIT_SHA}', script)
self.assertIn('if [[ "${CI_MODE}" == "merge" && "${BRANCH_NAME}" != "${RELEASE_ARTIFACT_BRANCH}" ]]', script)
self.assertIn('if [[ "${CI_MODE}" == "temp-image" && "${BRANCH_NAME}" == "${RELEASE_ARTIFACT_BRANCH}" ]]', script)
self.assertIn('SGL-Kernel wheel cache miss', kernel_build_script)
```

- [x] **Step 2: Run tests and confirm RED**

Run: `python3 test/jd-ci/unit/ci/test_internal_ci_contract.py -v`

Expected: 模式化缓存与 SGL-Kernel merge cache miss 契约失败。

- [x] **Step 3: Implement mode-scoped artifacts**

```bash
case "${CI_MODE}" in
    review|merge)
        WHEEL_CACHE_HOST="${PERSISTENT_SGL_KERNEL_CACHE_HOST}"
        MOONCAKE_ENGINE_CACHE_HOST="${PERSISTENT_MOONCAKE_ENGINE_CACHE_HOST}"
        MOONCAKE_STORE_WHEEL_CACHE_HOST="${PERSISTENT_MOONCAKE_STORE_WHEEL_CACHE_HOST}"
        ;;
    temp-image)
        JD_CI_TEMP_ARTIFACT_ROOT="${CI_ARTIFACT_ROOT}/tmp_artifacts/${COMMIT_SHA}"
        WHEEL_CACHE_HOST="${JD_CI_TEMP_ARTIFACT_ROOT}/sgl-kernel/${BASE_IMAGE_TAG}"
        MOONCAKE_ENGINE_CACHE_HOST="${JD_CI_TEMP_ARTIFACT_ROOT}/mooncake_te/${MOONCAKE_VERSION_TAG}"
        MOONCAKE_STORE_WHEEL_CACHE_HOST="${JD_CI_TEMP_ARTIFACT_ROOT}/mooncake_store/${MOONCAKE_VERSION_TAG}"
        ;;
esac
```

`-m` 使用 `MOONCAKE_REQUIRE_CACHE=1`；SGL-Kernel 在 `merge_request__merged` cache miss 时打印明确错误并退出，不进入现有 rebuild 命令。

- [x] **Step 4: Run tests and confirm GREEN**

Run: `python3 test/jd-ci/unit/ci/test_internal_ci_contract.py -v`

Expected: 正式与临时产物契约通过，原有三个组件构建调用次数不变。

### Task 3: 测试决策与两张镜像统一发布

**Files:**
- Modify: `test/jd-ci/unit/ci/test_internal_ci_contract.py`
- Modify: `test/jd-ci/run_jd_ci.sh`

**Interfaces:**
- Consumes: `RUN_CI_TESTS`、`PUBLISH_IMAGES`、`MAIN_EXIT_CODE`、`STORE_EXIT_CODE`。
- Produces: 三类回归报告或显式 skip 报告，以及正式/临时的两张镜像 tag。

- [x] **Step 1: Write the failing gate tests**

```python
self.assertIn("regression_report.py", script)
self.assertLess(script.index("STORE_EXIT_CODE="), script.index("docker commit"))
self.assertIn('if [[ ${EXIT_CODE} -eq 0 && "${PUBLISH_IMAGES}" == "1" ]]', script)
self.assertIn('_JD_TMP_${BRANCH_NAME_FOR_DOCKER}_${COMMIT_ID}', script)
```

契约还要断言三个回归 runner 在脚本中各出现一次，`-r` 不允许跳过测试，`JD_CI_SKIP_TEST=1` 只在 `-t` 生效。

- [x] **Step 2: Run tests and confirm RED**

Run: `python3 test/jd-ci/unit/ci/test_internal_ci_contract.py -v`

Expected: 临时测试决策、临时 tag 或统一发布顺序断言失败。

- [x] **Step 3: Implement test and publish gates**

```bash
if [[ "${RUN_CI_TESTS}" == "1" ]]; then
    jd_ci_failed=0
    bash "${SOURCE_PATH}/test/jd-ci/pipeline/run_cpu_mock_regression.sh" \
        "${SOURCE_PATH}" "${JD_CI_BASE_REF}" "${CI_LOGS_DIR}" || jd_ci_failed=1
    bash "${SOURCE_PATH}/test/jd-ci/pipeline/run_server_api_regression.sh" \
        "${SOURCE_PATH}" "${CI_LOGS_DIR}" || jd_ci_failed=1
    bash "${SOURCE_PATH}/test/jd-ci/pipeline/run_operator_regression.sh" \
        "${SOURCE_PATH}" "${CI_LOGS_DIR}" || jd_ci_failed=1
    python3 "${SOURCE_PATH}/test/jd-ci/report/generate_regression_summary.py" \
        --logs-dir "${CI_LOGS_DIR}" --event-type "${EVENT_TYPE}" \
        --branch "${BRANCH_NAME}" --commit "${COMMIT_ID}" \
        --base-image-tag "${BASE_IMAGE_TAG}" || jd_ci_failed=1
    exit "${jd_ci_failed}"
elif [[ "${CI_MODE}" == "temp-image" ]]; then
    for test_area in cpu_mock server_api operator; do
        python3 "${SOURCE_PATH}/test/jd-ci/report/regression_report.py" \
            --output "${CI_LOGS_DIR}/${test_area}/report.json" \
            --test-area "${test_area}" \
            --skip-reason "JD_CI_SKIP_TEST=1 in temp-image mode"
    done
    python3 "${SOURCE_PATH}/test/jd-ci/report/generate_regression_summary.py" \
        --logs-dir "${CI_LOGS_DIR}" --event-type "${EVENT_TYPE}" \
        --branch "${BRANCH_NAME}" --commit "${COMMIT_ID}" \
        --base-image-tag "${BASE_IMAGE_TAG}"
else
    echo '[JD CI] 正式合入模式: 跳过测试，只安装缓存并打包镜像'
fi
```

主容器和 Mooncake-store 容器运行完成后先汇总退出码；只有总退出码为 0 且 `PUBLISH_IMAGES=1` 时，才依次 commit 两个容器并 push 两张镜像。临时 tag 必须包含清洗后的分支名和短 commit。

- [x] **Step 4: Run tests and confirm GREEN**

Run: `python3 test/jd-ci/unit/ci/test_internal_ci_contract.py -v`

Expected: 测试与镜像门禁契约通过。

### Task 4: README、语法和完整契约验证

**Files:**
- Modify: `test/jd-ci/README.md`
- Modify: `test/jd-ci/temp_image_mode_design.md`
- Modify: `test/jd-ci/temp_image_mode_implementation_plan.md`

**Interfaces:**
- Consumes: 已落地的三种模式。
- Produces: 用户可直接复制的命令与最终实现状态记录。

- [x] **Step 1: Update documentation**

```text
-r / --review / note__merge_request
-m / --merge / merge_request__merged
-t / --temp-image
JD_CI_SKIP_SGL_KERNEL_BUILD=0|1
JD_CI_SKIP_MOONCAKE_BUILD=0|1
JD_CI_SKIP_TEST=0|1
```

README 必须明确正常 PR 不可跳过测试、正式合入 cache miss 失败、临时目录退出即清理、临时模式成功时产出两张镜像。

- [x] **Step 2: Run complete static verification**

Run:

```bash
bash -n test/jd-ci/run_jd_ci.sh test/jd-ci/env/build_sgl_kernel.sh
python3 -m unittest discover -s test/jd-ci/unit/ci -p 'test_*.py' -v
JD_CI_SERVER_API_DRY_RUN=1 bash test/jd-ci/pipeline/run_server_api_regression.sh "$PWD" /tmp/jd-ci-server-dry-run
JD_CI_OPERATOR_DRY_RUN=1 JD_CI_OPERATOR_AVAILABLE_GPUS=8 bash test/jd-ci/pipeline/run_operator_regression.sh "$PWD" /tmp/jd-ci-operator-dry-run
```

Expected: Shell 语法通过、全部 JD CI 契约通过、两类 GPU runner dry-run 通过，且没有 Docker commit/push。

- [ ] **Step 3: Review the final diff and commit**

```bash
git diff --check
git diff --name-only
git add test/jd-ci
git commit -m "ci: add explicit JD temporary image mode"
```

Expected: 只包含 `test/jd-ci/` 内的计划、脚本、契约与 README 改动。
