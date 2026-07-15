# JD SGLang CI

JD 内部分支的固定、累积回归门禁。主流水线保留容器、缓存、Mooncake 编译、
SGL-Kernel 编译、清理和镜像发布结构；评审事件只运行测试，不产出镜像。

## 如何选择入口

- 向 `JD-${BASE_IMAGE_TAG}` 提交正常 PR：使用 `-r`。每次重编两个组件、执行全部
  JD case、更新正式缓存，但不创建镜像。
- 任意分支需要快速产出验证镜像：使用 `-m`。只读复用对应版本主分支由 `-r`
  生成的正式缓存，不重编、不跑测试，cache miss 直接失败；成功后发布两张带当前
  commit 标识的镜像。
- 临时开发分支需要验证镜像：使用 `-t`。默认重编两个组件并执行全部 JD case；
  用户确认某组件无改动后，才可显式跳过该组件编译。成功后发布两张临时镜像。

无参数等价于 `-r`。旧调用方可继续使用 `note__merge_request` 和
`merge_request__merged`，分别等价于 `-r` 和 `-m`。

## 测试边界

- 只登记和执行 JD 内部 commit 对应的回归 case。
- JD CI 新增测试资产只能放在 `test/jd-ci/`；禁止在 `test/registered/`、
  `test/manual/` 或其他目录新增 JD 专属单测、benchmark、fixture、runner 和 helper。
- 不执行 SGLang 原生 `test/run_suite.py`、`test/registered` 或 `test/manual` 套件。
- case 可以复用开源代码和 helper，但断言必须指向 JD 引入、优化或修复的行为。
- 每次 CI 执行全部 active JD case，不根据当前 diff 动态选择或跳过测试组。
- 测试资产跨 JD 版本持续保留。只有 JD 改动被删除或被上游吸收时，才能带明确
  原因退役 case。

固定清单位于 `test/jd-ci/jd_test_manifest.py`。每项包含：

- 稳定 `case_id`；
- 完整 JD production commit SHA；由最终 JD CI commit 自身引入的 case 使用
  `tracks_ci_head=true`，避免在 commit 内自引用无法稳定的 SHA；
- `cpu`、`server`、`operator_correctness` 或 `operator_performance` 分类；
- 固定命令、GPU 数、超时和断言目标。

manifest contract 会拒绝未映射内部 commit、重复 case id、缺失文件以及原生测试
路径。Git diff 只允许用于发现新增 JD commit 是否缺少测试，不参与测试选择。

## CI 通过意味着什么

JD CI 通过表示：在当前固定清单、测试输入、GPU 型号和性能阈值内，没有发现 JD
内部改动造成的已覆盖功能、算子数值正确性或算子性能回归。它不是对所有模型、
所有输入和所有部署组合的无条件质量保证。

| 检查对象 | CI 通过可以证明 | CI 通过不能证明 |
| --- | --- | --- |
| CPU/mock | JD 协议解析、配置、状态转换和错误路径符合 case 断言 | 真实 Server、GPU 调度或真实模型输出正确 |
| Server/API | 单 Qwen2.5-VL dummy-weight Server 的文本/图像、流式/非流式、错误请求和 OpenAI 接口行为正常 | 真实 checkpoint 加载成功或模型生成精度不变 |
| 算子 correctness | 已登记 shape、dtype、边界和 dispatch 的结果符合参考实现 | 未登记输入空间以及完整模型链路都不存在数值问题 |
| 算子 performance | 同机同次运行的性能比值满足已登记阈值 | 所有机器、负载和模型上的端到端性能都不退化 |

因此，当前结果应表述为“已纳入 JD CI 清单的改动没有发现功能、算子正确性和
算子性能问题”，不能简化为“所有 JD 模型都没有精度问题”。如果一个 JD commit
改变真实权重加载、模型专属数值路径、调度/KV/分布式语义或端到端生成结果，必须
增加对应的最小真实 Server、模型或多卡 case，不能只依赖 mock 结论。

## 历史验证基线

### 当前单 Server 基线（2026-07-13）

在 `6.111.3.58:/export/zhangyu/sglang` 的 tmux 会话
`jd-ci-d2ae1ee12` 中完成提交 `d2ae1ee12` 的完整固定清单验证：

- JD CI 合同单测 `99/99` 通过，Shell 语法检查通过；
- CPU/Mock 回归 `6/6` 通过；
- Qwen2.5-VL dummy-weight Server 只启动 `1` 次，Server/API 子 case `15/15`
  通过；
- 算子 correctness/performance 回归 `8/8` 通过；
- 汇总报告为 `status=passed`、`failed_regressions=[]`；
- 日志已确认每 5 秒输出当前 case、已耗时和超时时间；
- 验证复用已有基础镜像，并安装当前 SGL-Kernel 缓存 wheel；没有构建、保存或推送
  新镜像。

报告目录：

```text
/export/zhangyu/ci/sglang/jd-ci/manual_validation/20260713_d2ae1ee12
```

### 旧聚合 Server 基线（2026-07-12）

2026-07-12 在 `6.111.3.58:/export/zhangyu/sglang` 完成了旧聚合 Server case 版本的
`codex/jd-ci-cumulative-regression` 的完整流水线验证：

- SGL-Kernel 编译 `379/379` 完成；
- CPU/Mock 回归 `6/6` 通过；
- Server/API 回归 `1/1` 通过；
- 算子正确性与性能回归 `8/8` 通过；
- 汇总报告为 `status=passed`、`failed_regressions=[]`；
- 评审事件未产出主镜像或 Mooncake-store 镜像。

完整 GPU 流水线对应功能提交 `6eedb906b`。最终提交 `628e4428d` 只清理残留术语
并补强命名契约，已在同一流水线机器重新通过 37 个 JD CI 单测、Shell 语法和命名
扫描。报告目录：

```text
/export/zhangyu/ci/sglang/jd-ci/ci_logs/20260711235352_codex-jd-ci-cumulative-regression_6eedb906b
```

## 三类固定回归

### CPU and Mock Regression：全部 JD CPU/mock 回归

强制隐藏 GPU，执行 manifest 中全部 `cpu` case。覆盖：

- OpenAI 协议、thinking/ignore-EOS 和函数调用解析；
- tokenizer/request state、多模态清理和超时；
- EPLB/DP metadata、CUDA Graph metadata、W4A8 配置；
- L1/L2 cache metrics；
- JD deploy/TMA 配置；
- JD CI 编译、缓存、日志和发布契约。

不会调用 `test/run_suite.py`。

```bash
bash test/jd-ci/pipeline/run_cpu_mock_regression.sh "$PWD" v0.5.15 /tmp/jd-ci
```

### Server and API Regression：全部 JD Server/API 回归

仅暴露一张 GPU，固定使用
`/mnt/nas/models/Qwen2.5-VL-7B-Instruct/` 的 tokenizer/config，以
`--load-format dummy --disable-cuda-graph --tp-size 1` 启动一次真实 SGLang Server。
Server/API 回归不引入上游 mock-model 的 `token_oracle`、KV canary 等非 JD 验证项；
CUDA Graph 的 JD 修复由 CPU fixture 独立覆盖。整个回归只能存在一个
`popen_launch_server` 调用，不为模型特定修复重复启动 GLM、Kimi 或 DeepSeek Server。

同一个进程固定执行 15 个可见 HTTP 子 case：

1. `/v1/models`；
2. 文本非流式与流式请求；
3. 内联图片非流式与流式请求；
4. 损坏 base64、不可达图片 URL 的非流式和流式错误请求，要求返回客户端错误而非
   HTTP 500；
5. `ignore_eos` 使用 JD 默认上限生成 8 个 token，并以 `length` 结束；
6. list、缺少 `type` 的 dict、未知 string、integer 四类 invalid `thinking`；
7. `tool_choice=none`。

模型特定但在 tokenizer/model inference 前已经完成的修复，使用 JD 自有确定性 CPU
fixture：DeepSeek-V4 reasoning 开关、GLM reasoning parser 的真实字符串解析、Kimi-K2
带引号参数、DSV4 非法参数编码和 `reasoning_tokens` 统计。只有风险依赖 checkpoint、
tensor、logits、数值输出或生成语义时，才增加最小真实模型 case。

它验证真实 HTTP/request/lifecycle 路径，不声称验证真实 checkpoint 加载或生成精度。

```bash
bash test/jd-ci/pipeline/run_server_api_regression.sh "$PWD" /tmp/jd-ci
```

### Operator Correctness and Performance Regression：全部 JD 算子功能和性能回归

每次执行 manifest 中全部 `operator_correctness` 和 `operator_performance` case。
每个算子必须成对登记：

- correctness 与参考实现比较代表性 shape、dtype、边界和 dispatch；
- performance 在同一流水线、同一次运行内完成 warm-up 和重复采样，对比参考路径，
  避免使用跨机器绝对延迟阈值。

单个算子 case 失败时会立即记录失败，但 runner 继续执行其余固定 case；全部 case
结束后统一返回非零状态，确保一次 CI 能收集完整的算子回归结果。

当前固定覆盖 optimized RMSNorm、DP-attention compressed all-gather、DSV4
norm-rope 和 W4A8 dynamic quantization。

```bash
bash test/jd-ci/pipeline/run_operator_regression.sh "$PWD" /tmp/jd-ci
```

## 实时日志协议

三类 runner 都在执行前打印完整固定清单。每个真实 case 输出 `START`，运行中由
`test/jd-ci/pipeline/case_progress.py` 每 5 秒输出一次当前 case、当前动作、已耗时和
超时时间，结束时输出 `PASS` 或 `FAIL`。硬件不足输出 `BLOCKED`，显式 dry-run 输出
`SKIP`。Server 启动另用 `SERVER` 行展示启动和健康检查进度。

```text
[JD CI][Server and API Regression][INVENTORY 3/15] id=jd-text-streaming ... timeout=60s
[JD CI][Server and API Regression][CASE 3/15][START] id=jd-text-streaming ...
[JD CI][Server and API Regression][CASE 3/15][RUNNING] id=jd-text-streaming phase=http-request elapsed=5.0s timeout=60s
[JD CI][Server and API Regression][CASE 3/15][PASS] id=jd-text-streaming duration=6.2s exit_code=0
```

CPU/Mock、Server/API 和算子 runner 遇到普通 case 失败后都继续执行剩余固定 case，
最后统一返回非零状态。每个 JSON case 保留 `assertion`、`duration_seconds`、
`timeout_seconds`、`exit_code`、`detail` 和 `log_file`，不再要求等待整组结束后才能知道
测试到了哪里、测了什么以及结果如何。

## 主流水线

```bash
bash test/jd-ci/run_jd_ci.sh -h  # 查看完整帮助
bash test/jd-ci/run_jd_ci.sh -r  # 默认：正常 PR 评审
bash test/jd-ci/run_jd_ci.sh -m  # 任意分支复用正式缓存快速产出镜像
bash test/jd-ci/run_jd_ci.sh -t  # 临时分支验证镜像
```

| 入口 | 组件产物 | 固定累积 JD 回归 | 镜像 |
| --- | --- | --- | --- |
| `-r` / `--review` / `note__merge_request` / 无参数 | 强制编译并更新正式缓存 | 固定全量执行 | 不产出 |
| `-m` / `--merge` / `merge_request__merged` | 任意分支只安装对应版本主分支的正式缓存，cache miss 失败 | 固定跳过 | 产出带当前 commit 标识的 SGLang 和 Mooncake-store 镜像 |
| `-t` / `--temp-image` | 继承基础镜像或在 commit 临时目录编译 | 默认全量执行，可显式全部跳过 | 产出与 `-m` 标签格式一致的 SGLang 和 Mooncake-store 镜像 |

`-r` 固定按 CPU/Mock、Server/API、算子正确性与性能的顺序执行全部三类回归。
任一回归失败会记录失败状态，但不会阻止后续回归执行；三类回归结束后统一生成报告
并返回失败。`-r` 不存在回归跳过开关，也不允许按 commit diff 跳过 case。

`-m` 允许任意分支只读复用对应版本主分支的正式 SGL-Kernel、Mooncake TE 和
Mooncake-store 缓存；任一 cache miss 都直接失败，不允许回退到源码编译。它固定
跳过 JD 回归，并产出带当前 commit 标识的两张镜像。`note__merge_request` 和
`merge_request__merged` 继续分别兼容 `-r` 和 `-m`，方便现有流水线事件调用方
无缝迁移。

`-t` 只能在非正式分支运行，三个临时选项只接受 `0` 或 `1`：

| 环境变量 | 默认值 | `0` | `1` |
| --- | --- | --- | --- |
| `JD_CI_SKIP_SGL_KERNEL_BUILD` | `0` | 在本次 commit runner 的 `artifacts/` 目录编译并安装 | 用户确认无相关改动，继承基础镜像组件 |
| `JD_CI_SKIP_MOONCAKE_BUILD` | `0` | 在同一 commit 临时根目录编译并安装 TE/store | 用户确认无相关改动，继承基础镜像组件 |
| `JD_CI_SKIP_TEST` | `0` | 固定执行全部三类 JD 回归 | 跳过全部回归并生成显式 skip 报告 |

三个变量默认均为 `0`，并且只允许用于 `-t`。任何一个变量在 `-r` 或 `-m` 中设为
`1` 都会在 Git/Docker 主流程前以状态码 `2` 拒绝。`JD_CI_SKIP_TEST=1` 只表示用户
主动放弃本次临时镜像的回归结论，不代表测试通过；汇总报告会把三类回归都记录为
显式 skip。

临时模式不读取、写入或覆盖正式组件缓存。成功、失败、中断和镜像推送完成后都会
清理 commit 临时目录。主容器与 Mooncake-store 容器被视为同一组验证产物；任一
组件、容器或已启用测试失败时，两张镜像都不推送。全部门禁通过后才产出镜像；
`-t` 与 `-m` 统一使用 `*_JD_${COMMIT_ID}` 标签格式。

```bash
# 组件都编译并执行全部测试
bash test/jd-ci/run_jd_ci.sh -t

# 用户确认 SGL-Kernel 无改动，只继承基础镜像中的版本
JD_CI_SKIP_SGL_KERNEL_BUILD=1 bash test/jd-ci/run_jd_ci.sh -t

# 用户确认两个组件均无改动，并显式跳过全部 JD 回归
JD_CI_SKIP_SGL_KERNEL_BUILD=1 \
JD_CI_SKIP_MOONCAKE_BUILD=1 \
JD_CI_SKIP_TEST=1 \
  bash test/jd-ci/run_jd_ci.sh -t
```

常见临时分支选择：

| 分支改动 | 推荐命令前缀 | 实际行为 |
| --- | --- | --- |
| SGLang 或 SGL-Kernel 有改动，Mooncake 无改动 | `JD_CI_SKIP_MOONCAKE_BUILD=1` | 重编 SGL-Kernel，继承基础镜像 Mooncake，执行全量回归 |
| Mooncake 有改动，SGL-Kernel 无改动 | `JD_CI_SKIP_SGL_KERNEL_BUILD=1` | 重编 Mooncake TE/store，继承基础镜像 SGL-Kernel，执行全量回归 |
| 两个组件均无改动 | 两个组件 skip 均为 `1` | 继承基础镜像组件，仍执行全量回归 |
| 只需验证镜像能否打包 | 组件 skip 按实际改动设置，并加 `JD_CI_SKIP_TEST=1` | 不执行回归，生成显式 skip 报告后再经过双容器门禁 |

## 缓存、报告与清理

默认 `CI_WORK_DIR=/export/zhangyu`，各类产物位置如下：

| 内容 | 路径 | 生命周期 |
| --- | --- | --- |
| SGL-Kernel 正式缓存 | `${CI_WORK_DIR}/ci/sglang/sgl-kernel/${BASE_IMAGE_TAG}` | `-r` 更新，`-m` 只读 |
| Mooncake TE 正式缓存 | `${CI_WORK_DIR}/ci/sglang/mooncake_te/${MOONCAKE_VERSION_TAG}` | `-r` 更新，`-m` 只读 |
| Mooncake-store 正式缓存 | `${CI_WORK_DIR}/ci/sglang/mooncake_store/${MOONCAKE_VERSION_TAG}` | `-r` 更新，`-m` 只读 |
| 本次 runner 日志、临时依赖和 `-t` 产物 | `${CI_WORK_DIR}/ci/sglang/jd-ci/runners/${COMMIT_ID:0:9}` | 仅本次 CI 存在，退出时整体清理 |

本次 runner id 固定为 commit id 的前 9 位。runner 目录内按用途隔离：

```text
runners/<9-char-commit>/
├── logs/
│   ├── containers/       # 主 SGLang 容器、mooncake-store 容器
│   ├── builds/           # SGL-Kernel、Mooncake TE/store 编译日志
│   └── tests/            # CPU/Mock、Server/API、Operator 独立 case 日志和报告
├── work/
│   ├── containers/       # 两个容器各自的临时目录和缓存
│   ├── builds/           # 三类编译各自的依赖和中间文件
│   └── tests/            # 三类回归各自的临时依赖和缓存
└── artifacts/                 # 仅 `-t` 使用的临时 wheel
```

失败时会在删除前把完整日志转储到流水线 stdout。退出 trap 先停止
Docker CLI 并删除两个容器，再删除整个 runner 目录。无论成功、失败还是中断，
日志、报告、临时依赖、编译中间文件和临时产物都不会在磁盘上持续保留。

## 提交前与流水线机器验证

本地或 CPU 机器先执行静态契约和 GPU runner dry-run，不会创建镜像：

```bash
bash -n test/jd-ci/run_jd_ci.sh test/jd-ci/env/*.sh test/jd-ci/pipeline/*.sh
python3 -m unittest discover -s test/jd-ci/unit/ci -p 'test_*.py' -v
JD_CI_SERVER_API_DRY_RUN=1 \
  bash test/jd-ci/pipeline/run_server_api_regression.sh "$PWD" /tmp/jd-ci-server
JD_CI_OPERATOR_DRY_RUN=1 JD_CI_OPERATOR_AVAILABLE_GPUS=8 \
  bash test/jd-ci/pipeline/run_operator_regression.sh "$PWD" /tmp/jd-ci-operator
```

完整验证必须在流水线机器的干净工作区运行：

```bash
cd /export/zhangyu/sglang
git status --porcelain  # 必须无输出
tmux new-window -t agent -n jd-ci-review \
  'cd /export/zhangyu/sglang && bash test/jd-ci/run_jd_ci.sh -r'
```

只验证代码和测试时使用 `-r`；不要执行完整 `-m` 或 `-t`，因为这两个模式成功时
会创建并推送镜像。`-m` 已允许任意分支运行，不能再用分支保护做无镜像验证；只在
用户明确授权发布当前 commit 镜像后执行。

## JD skill 受控自进化

`$xq-sglang-jd-new-pr` 和 `$xq-sglang-jd-release-rebase` 使用“稳定核心 + 独立规则层”。
`SKILL.md` 和 JD CI 硬边界保持稳定，每个 skill 在自己的 `evolution/` 目录记录证据，
不能跨 skill 复用未审核规则。任务开始先执行：

```bash
python3 test/jd-ci/jd_skill_evolution.py check --skill-dir <skill-dir>
python3 test/jd-ci/jd_skill_evolution.py list --skill-dir <skill-dir> --status promoted
```

低风险加法规则必须在两次独立任务和不同源码上下文中重复出现，通过规则单测且没有
反例后才能自动晋升。语义等价、JD 修复或 case 退役、精度与性能阈值、模型/GPU
范围、冲突业务行为和发布策略必须人工审核。进化规则只能增加检查、覆盖或诊断，
不能削弱 JD CI，不能改变固定累积全量执行，也不能把真实模型、真实 Server、多卡或
算子正确性与性能测试降级为 mock。

Release Rebase skill 可以自动积累机械路径/API 迁移、确定性符号改名和验证命令。
语义等价不能自动判定；删除 JD 内部修复或 case 必须人工确认。学习到的规则只能
帮助定位上游证据，不能自行调用 `classify` 或把内部 commit 标记为
`absorbed-semantic`。

## 新增或修改 JD commit

推荐调用 `$xq-sglang-jd-new-pr`，先对比当前分支与目标 JD 分支，再自动生成
commit-to-case 清单、补充最小 JD 单测并更新固定 manifest。必须显式确认正确的 PR
目标分支；diff 只用于发现缺失 case，不改变 CI 固定全量执行行为。

```bash
python3 .agents/skills/xq-sglang-jd-new-pr/scripts/collect_pr_delta.py \
  --repo "$PWD" --base <JD-target-ref> --output /tmp/jd-pr-delta.json
```

1. 在 manifest 增加 commit SHA 和 case。
2. Server/model 逻辑若在 inference 前完成，优先写确定性 CPU mock/parser fixture；
   必须验证 HTTP/lifecycle 时向单 Qwen2.5-VL dummy Server 的 `SERVER_CASES` 增加子
   case，禁止新增第二个 Server。
3. 真实 checkpoint、数值精度或模型-算子集成无法由 mock 证明时，增加最小真实模型
   case，而不是恢复大模型粒度通用精度评测。
4. 算子新增或优化必须同时提交 correctness 和 performance case。
5. 本地先运行 manifest、CPU/Mock 回归以及 Server/API 和算子回归的 dry-run，
   再在流水线机器执行完整 GPU 回归。

```bash
python3 test/jd-ci/jd_test_manifest.py \
  --source "$PWD" --output /tmp/jd-cases.json
JD_CI_SERVER_API_DRY_RUN=1 \
  bash test/jd-ci/pipeline/run_server_api_regression.sh "$PWD" /tmp/jd-ci
JD_CI_OPERATOR_DRY_RUN=1 JD_CI_OPERATOR_AVAILABLE_GPUS=8 \
  bash test/jd-ci/pipeline/run_operator_regression.sh "$PWD" /tmp/jd-ci
```

## 开源新版本发布后的 JD 迁移

调用 `$xq-sglang-jd-release-rebase`，从新开源 tag 创建新的 JD 分支，按原始顺序
重放仍为 JD 专属的 commit，并自动维护冲突状态、旧新 SHA 映射和 manifest 迁移。
旧 JD 分支保持不变；已被上游吸收的 patch 不再作为 JD 内部 case 保留。

核心原则：外部已有同类修复，放弃内部修复；外部没有同类修复，继续使用内部修复。
同类修复按问题和最终行为判断，不只比较 patch-id。实现不同但语义等价时，用
`classify` 将内部 commit 标记为 `absorbed-semantic` 并记录外部修复依据。

先生成只读计划：

```bash
python3 .agents/skills/xq-sglang-jd-release-rebase/scripts/jd_release_rebase.py plan \
  --repo "$PWD" \
  --old-internal refs/remotes/origin/JD-v0.5.14 \
  --old-upstream refs/tags/v0.5.14 \
  --new-upstream refs/tags/v0.5.15 \
  --new-internal JD-v0.5.15 \
  --output /tmp/jd-release-rebase-plan.json
```

确认计划中的完整 refs、commit 顺序、absorbed patch 和高风险文件后，再由 skill 执行
`execute`、冲突处理和 `resume`。生产提交重放完成后，使用 `prepare-ci` 合并应用全部
历史 JD CI 改动但不产生中间 commit；完成 manifest、构建兼容、文档、测试和两个
JD skill 的迁移后，使用 `commit-ci` 创建唯一 JD CI commit。该 commit 必须位于分支
`HEAD`，之后禁止再追加 production、兼容、元数据或文档 commit。

SGL-Kernel 和 Mooncake 的编译前依赖每次升级都要按新版本源码重新审计。控制机可以
联网获取上游指定 tag/SHA；v0.5.15 的 SGL-Kernel FetchContent 统一使用容器内
`/wheels/_deps`，映射到控制机
`/export/zhangyu/ci/sglang/sgl-kernel/v0.5.15/_deps`，不能复用旧版本目录作为新版本
事实来源。该目录持久化下载压缩包和 `*-src`；编译前会清理 `*-build` 以及
`*-subbuild` 中绑定旧容器绝对路径的 CMake 状态，避免缓存迁移后触发路径冲突，且
不会删除已下载依赖。Mooncake Store 继续使用 JD 内部镜像仓库；内部包缺失时停止并报告，禁止
切换公共源规避问题。

最后运行 `check`，要求旧新 SHA 映射完整、manifest 无旧 SHA、JD CI commit 数量为
1 且就是 `HEAD`。默认只创建本地分支和 worktree，不 push、不启动远端流水线、不
产出镜像。迁移后还必须复核单 Server 的 15 项固定清单、模型特定 CPU fixture、
5 秒心跳和普通 case 失败继续行为没有被上游接口变化破坏。
