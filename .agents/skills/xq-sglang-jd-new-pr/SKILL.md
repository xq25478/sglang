---
name: xq-sglang-jd-new-pr
description: 用于审查新的 JD SGLang PR 或内部开发分支；对比目标 JD 发布分支，分析每个 JD commit，并在 CI 前补齐 JD 专属回归测试。
---

# JD SGLang 新 PR 回归测试

## 目标

把当前分支相对目标 JD 分支的每个新增 commit，转换为可审计的 commit-to-case 映射，并补充最小但充分的 JD 专属测试。

本 Skill 只测试 JD 引入、优化或修复的行为，不执行上游原生测试清单。每次 CI 都执行固定累积全量清单，不根据当前 diff 动态选择 case。

## 强制边界

- 修改前必须确定明确 base、merge-base、commit 范围和工作树状态；base 有歧义时停止。
- 所有新增 JD 测试资产必须位于 `test/jd-ci/`。单测、benchmark、fixture、runner、报告脚本和 test-only helper 都不得新增到其他目录。
- 保留用户已有的 tracked 和 untracked 改动，只修改当前任务需要的文件。
- 未经单独授权，不得 commit、push、发布镜像或启动远端 CI。
- 跳过 GPU 执行只能记录为证据缺失，不能报告为通过。
- 每个普通 PR commit 必须映射到可观察断言，或明确说明阻塞原因。
- 最终 JD CI commit 自身新增的 case 使用 `tracks_ci_head=true`，不得尝试在该 commit 中写入自身 SHA。

## 受控自进化

先检查规则状态：

```bash
python3 test/jd-ci/jd_skill_evolution.py check \
  --skill-dir .agents/skills/xq-sglang-jd-new-pr
python3 test/jd-ci/jd_skill_evolution.py list \
  --skill-dir .agents/skills/xq-sglang-jd-new-pr --status promoted
```

`evolution/policy.json` 是稳定策略。自动规则引擎不得改写 `SKILL.md`，不得 commit 或 push，不得启动 CI 或发布镜像，也不得用 mock 覆盖替代 real-server、real-model、多 GPU、算子正确性或算子性能覆盖。

低风险的路径/符号映射、验证补充、失败特征和加法覆盖模式，只有在两个独立任务、不同源码上下文中重复出现并通过规则测试后才能晋升。语义等价、覆盖降级、case 退役、精度或性能阈值、模型/GPU 范围、硬边界和发布策略必须进入 `pending-review` 人工审核。

需要记录经验时：

```bash
python3 test/jd-ci/jd_skill_evolution.py record --skill-dir <skill-dir> ...
python3 test/jd-ci/jd_skill_evolution.py evaluate --skill-dir <skill-dir> ...
python3 test/jd-ci/jd_skill_evolution.py promote --skill-dir <skill-dir> ...
```

最终任务总结包含“进化报告”，列出命中的规则、新候选、证据数量和人工审核项。

## 1. 读取仓库约束

完整阅读：

- `test/jd-ci/README.md`
- `test/README.md`
- `test/jd-ci/jd_test_manifest.py`
- `test/jd-ci/operator_registry.py`
- 当前改动路径附近的 `AGENTS.md`

记录：

```bash
git status --short
git branch --show-current
git rev-parse HEAD
git worktree list
```

## 2. 确定对比分支

优先使用用户显式提供的 JD 目标 ref。未提供时，只能选择唯一且明确的最近 JD 发布祖先，不能用 `origin/HEAD` 替代。

运行：

```bash
python3 .agents/skills/xq-sglang-jd-new-pr/scripts/collect_pr_delta.py \
  --repo "$PWD" --base <JD-target-ref> --output /tmp/jd-pr-delta.json
```

确认输出中的 `resolved_base`、`git merge-base`、`merge_base..HEAD` commit 列表、文件状态和 `uncovered_commits`。detached HEAD、无效 ref 或 base 有歧义时停止。

## 3. 审查每个新增 commit

对每个 uncovered commit 检查：

```bash
git show --stat --summary <sha>
git show --find-renames --find-copies <sha>
git diff <merge-base>...HEAD -- <path>
rg -n "<symbol-or-behavior>" python sgl-kernel test/jd-ci
```

不要只看 subject 或文件名。确认真实调用路径、输入输出、不变量、失败模式、硬件要求，以及上游是否已有相同测试。

### 审计 PR 已有覆盖

- 先审计 PR 已有测试资产，包括新加或修改的 test、benchmark、fixture 和注册逻辑；逐项判断是否命中本次改动，而不是默认直接采用。
- 把每个 case 的输入与新增 guard、feature gate 和真实 dispatch 条件逐项对照。至少一个 case 必须经过生产调用入口，并用 spy、计数器或其他可观察证据证明命中了目标分支；只直接调用底层 helper 不足以证明接线正确。
- 改动共享 wrapper、导出符号或公共函数签名时，必须对比 base 公共接口并搜索所有生产调用点；为仍在使用的 scoring mode、grouped 参数、dtype、环境开关和返回约定补兼容性 canary。
- 性能测试的 candidate 与 reference 必须进入不同的底层实现。沿调用图确认实际 kernel；两个 wrapper 最终落到同一 kernel 的自比 benchmark 无效，必须换成独立实现或明确阻塞。
- PR 已把 JD-only 文件放到 `test/jd-ci/` 之外时，复用其中有效逻辑后，迁移或删除原位置的重复 JD 测试资产；不得留下重复注册或用目录外测试冒充 JD case。

文档或纯测试 commit 可以明确排除；其余 commit 必须进入映射。

## 4. 选择最小充分测试

| 改动类型 | 首选验证 | 放置位置 |
| --- | --- | --- |
| 解析、校验、配置、状态转换、错误路径 | CPU/mock | `test/jd-ci/unit/<area>/test_*.py` |
| OpenAI HTTP、流式、endpoint、request lifecycle | dummy-weight 真实 Server | `test/jd-ci/pipeline/` |
| CUDA/C++/Triton/JIT/AOT 算子 | correctness + performance 成对 | `test/jd-ci/operators/` |
| checkpoint 加载、真实生成、模型-算子集成 | 最小 real-model case | `test/jd-ci/` |
| 多卡调度、通信、KV/缓存语义 | 最小多 GPU case | `test/jd-ci/` |
| 构建、缓存、打包、环境行为 | shell/Python contract | `test/jd-ci/unit/ci/` |

### 算子要求

每个算子必须同时存在 `operator_correctness` 和 `operator_performance`，两者使用相同 `operator` 和 commit 集合。性能 case 使用相对基线、预热、重复测量，并记录硬件和容差。

### 保持单 Server 设计

Server/API 回归只能启动一个 `Qwen2.5-VL-7B-Instruct/` dummy-weight Server。需要新增 HTTP 覆盖时扩展 `SERVER_CASES`，不得启动第二个 Server。

必须保留：

- 15 个固定 HTTP 子 case；
- `--disable-cuda-graph`；
- 文本/图像、流式/非流式、错误请求和 OpenAI 接口；
- `token_oracle` 与 KV canary；
- 上游 mock-model 能力存在时优先使用；模型特定 parser 若不能由该 Server 真实触发，放入确定性 CPU fixture；
- `case_progress.py` 每 5 秒心跳；
- 执行全部固定 JD case；
- 普通 case 失败后继续执行，最终统一失败。

mock 只能证明进入真实推理前可观察的确定性逻辑，不能代替真实 checkpoint、真实数值路径或多卡语义。

## 5. 先写失败测试

使用 TDD：

1. 先写最小断言；
2. 在缺少修复或映射的状态下观察 RED；
3. 再补充测试资产或 manifest 映射；
4. 运行聚焦测试得到 GREEN；
5. 保留能证明断言有效的失败证据。

不得为了通过测试而扩大生产代码改动。

## 6. 更新固定 manifest

在 `test/jd-ci/jd_test_manifest.py` 的 `INTERNAL_COMMITS` 中登记普通 JD production commit 的完整 SHA，并新增或扩展 `JDCase`：

- `case_id` 稳定且描述性强；
- `commits` 使用真实 JD commit；
- category 必须是 `cpu`、`server`、`operator_correctness` 或 `operator_performance`；
- command 只能指向 `test/jd-ci/`；
- 写清 assertion、GPU 数和 timeout；
- 最终单一 JD CI HEAD commit 引入的 CI 合同或算子 case 使用空 `commits` 与 `tracks_ci_head=true`。

正常新增算子不需要修改 `operator_registry.py`；只有 registry 行为本身变化时才修改。

## 7. 本地验证

```bash
python3 test/jd-ci/jd_test_manifest.py \
  --source "$PWD" --output /tmp/jd-cases.json
python3 -m unittest discover -s test/jd-ci/unit/ci -p 'test_*.py' -v
JD_CI_CPU_MOCK_DRY_RUN=1 \
  bash test/jd-ci/pipeline/run_cpu_mock_regression.sh "$PWD" <base-ref> /tmp/jd-ci
JD_CI_SERVER_API_DRY_RUN=1 \
  bash test/jd-ci/pipeline/run_server_api_regression.sh "$PWD" /tmp/jd-ci
JD_CI_OPERATOR_DRY_RUN=1 JD_CI_OPERATOR_AVAILABLE_GPUS=8 \
  bash test/jd-ci/pipeline/run_operator_regression.sh "$PWD" /tmp/jd-ci
```

验证新增文件没有越界：

```bash
git diff --diff-filter=A --name-only "$BASE_REF"...HEAD
```

任何新增 JD test-only 文件不以 `test/jd-ci/` 开头都必须移动。

## 8. 流水线机器真实验证

本地合同检查完成后，必须到用户指定的流水线机器执行与改动匹配的真实验证；这是完成条件，不是可选建议。

- 使用独立目录和 tmux 会话，不能污染正式源码目录。
- 先记录远端 commit、镜像、CUDA、GPU、缓存目录和命令。
- 新 PR 默认运行固定全部 case；GPU 不足或依赖缺失时明确报告阻塞。
- 除非用户单独授权，验证不得保存、发布或推送镜像。
- `-m` 允许任意分支只读复用对应版本主分支的正式缓存，固定跳过测试并快速产出
  当前 commit 镜像；任一 cache miss 都立即失败，不得回退源码编译。镜像发布仍需用户单独授权。
- 需要临时测试镜像时，只能按 `-t` 规则执行，并遵守用户明确的镜像授权。

## 9. 完成检查

确认：

- base、merge-base、HEAD 和 commit 范围可审计；
- 每个生产 commit 有真实 case 映射或明确阻塞；
- `uncovered_commits` 只剩明确排除的文档/纯测试 commit；
- 固定累积全量清单没有被 diff 选择逻辑替代；
- 算子 correctness/performance 成对；
- 单 Server、15 个子 case、5 秒心跳和失败继续行为保留；
- 所有 JD 测试资产位于 `test/jd-ci/`；
- 本地与流水线机器证据均已记录；
- 未经授权没有 commit、push、启动镜像发布或产生镜像。

最终报告列出 commit-to-case 映射、修改文件、RED/GREEN 证据、未执行项、流水线机器结果和进化报告。
