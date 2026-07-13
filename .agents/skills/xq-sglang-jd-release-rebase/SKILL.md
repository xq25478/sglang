---
name: xq-sglang-jd-release-rebase
description: 用于开源 SGLang 发布新版本后，将现有 JD 发布线迁移到新上游 tag，重放内部修复、处理冲突、迁移 JD CI 并完成验证。
---

# JD SGLang 开源版本升级

## 目标

在新的开源 tag 上创建全新的 JD 分支，按原始顺序重放仍然需要的内部 commit，保持旧 JD 分支不变，并维护完整的 old-to-new SHA 映射。

必须同时调用 `$xq-sglang-jd-new-pr`，审查升级中新增的兼容改动、冲突适配和缺失测试。

## 核心原则

- 外部已有同类修复：放弃内部修复。
- 外部没有同类修复：继续使用内部修复。
- “同类修复”按解决的问题和最终行为判断，不能只比较 subject、文件名或 patch-id。
- 语义等价必须有源码、调用路径和测试证据；证据不足时保留内部修复。
- 用户对 `classify`、依赖选择、case 退役和发布动作拥有最终决定权。

## 强制边界

- 使用完整 `refs/tags/`、`refs/heads/` 或 `refs/remotes/`，写操作前解析到完整 SHA。
- 新分支必须在 isolated worktree 中创建，旧 JD 分支不可修改。
- 保留仍属 JD 专有的 commit 原始顺序，所有重写 SHA 写入状态文件。
- JD CI commit boundary 包含：
  - `test/jd-ci/`
  - `.agents/skills/xq-sglang-jd-new-pr/`
  - `.agents/skills/xq-sglang-jd-release-rebase/`
- 任何触及上述路径的 commit 整体延后；mixed production and JD CI commit 也不得拆分。
- 所有延后历史、manifest 迁移、构建兼容、文档、测试和两个 Skill 更新，必须合并为 exactly one commit，并且 must be `HEAD`。之后不得追加任何 production、兼容、元数据或文档 commit。
- 所有新增 JD test-only 资产只能位于 `test/jd-ci/`。
- 未经单独授权，不得 push、force-push、删除分支、启动远端 CI 或发布镜像；重写已发布分支只能使用 `--force-with-lease`。
- 冲突存在多个合理业务结果、依赖存在多个可行版本或需要降低覆盖时，停止并询问用户。

## 受控自进化

先运行：

```bash
python3 test/jd-ci/jd_skill_evolution.py check \
  --skill-dir .agents/skills/xq-sglang-jd-release-rebase
python3 test/jd-ci/jd_skill_evolution.py list \
  --skill-dir .agents/skills/xq-sglang-jd-release-rebase --status promoted
```

`evolution/policy.json` 是稳定策略。自动规则引擎不得改写 `SKILL.md`，不得 commit 或 push，不得启动 CI 或发布镜像，不得自动判定内部修复已被吸收，也不得自动调用 `classify`。

只有机械路径/符号映射、验证补充、失败特征和确定性 API 改名适配可在两次独立证据后自动晋升。语义等价、`absorbed-semantic`、内部修复退役、case 退役、冲突业务行为、依赖选择、精度/性能阈值、模型/GPU 范围、硬边界和发布策略必须进入 `pending-review`。

需要记录经验时：

```bash
python3 test/jd-ci/jd_skill_evolution.py record --skill-dir <skill-dir> ...
python3 test/jd-ci/jd_skill_evolution.py evaluate --skill-dir <skill-dir> ...
python3 test/jd-ci/jd_skill_evolution.py promote --skill-dir <skill-dir> ...
```

最终报告包含“进化报告”。

## 1. 升级前检查

记录：

```bash
git status --short
git branch --show-current
git rev-parse HEAD
git worktree list
git remote -v
```

确认没有未完成的 cherry-pick、rebase 或 merge，并保护 unrelated tracked/untracked changes。

获取对象：

```bash
git fetch <upstream-remote> --tags --prune
git fetch <internal-remote> --prune
```

控制机可以联网。允许使用网络获取精确上游 tag、依赖源码和权威版本信息；网络权限不等于 push、启动远端 CI 或发布镜像授权。

示例 refs：

```text
old upstream: refs/tags/v0.5.14
old JD:       refs/remotes/origin/JD-v0.5.14
new upstream: refs/tags/v0.5.15
new JD:       refs/heads/JD-v0.5.15
```

记录四个完整 SHA，确认新分支和目标 worktree 不存在。

## 2. 审计编译前依赖

Build Prerequisite Decision 必须在 `execute` 前完成。对比旧 tag、新 tag 和旧 JD 分支：

```bash
git diff refs/tags/<old>..refs/tags/<new> -- \
  sgl-kernel/pyproject.toml \
  sgl-kernel/CMakeLists.txt \
  sgl-kernel/cmake \
  docker/Dockerfile \
  scripts/ci/cuda/ci_install_dependency.sh

git diff refs/tags/<old>..refs/remotes/<internal>/<old-jd> -- \
  test/jd-ci/env/build_sgl_kernel.sh \
  test/jd-ci/env/build_mooncake.sh \
  test/jd-ci/run_jd_ci.sh
```

还要审查下载后的依赖 revision、submodule、编译选项和 release notes。

### SGL-Kernel

必须核对：

- Python/PyTorch ABI；
- CUDA/NVCC 和目标 SM；
- CMake/Ninja/scikit-build；
- FlashAttention/FlashMLA/CUTLASS；
- 新增扩展、头文件、生成代码和 wheel 布局；
- 架构开关、并行参数和缓存键。

目标版本已由上游唯一指定时，不再询问版本选择：获取该版本，使用 immutable tag or commit SHA，并更新 dependency declaration 和 lock files。仍有多个可行版本时必须获得用户明确批准。

SGL-Kernel 依赖必须预取到控制机版本隔离目录：

```text
/export/zhangyu/ci/sglang/sgl-kernel/<target-version>/_deps
```

本次 v0.5.15 使用：

```text
/export/zhangyu/ci/sglang/sgl-kernel/v0.5.15/_deps
```

不得把旧版本 `_deps` 当作新版本事实来源。构建脚本必须消费该目录，并记录每个解析后的 tag/SHA。
持久化缓存只把下载压缩包和 `*-src` 作为可跨容器复用的事实；`*-build` 和
`*-subbuild` 中的 CMake 状态可能记录旧容器绝对路径。正式编译前必须清除这些
可重建状态，同时保留压缩包和源码，禁止因清理缓存而重新访问外网。

### Mooncake

从 `docker/Dockerfile` 和 `scripts/ci/cuda/ci_install_dependency.sh` 解析安装版本，再由 `test/jd-ci/env/build_mooncake.sh` 解析到精确 JD Mooncake branch/commit。

核对：

- RDMA/libibverbs；
- NUMA；
- protobuf/gRPC；
- Rust/cargo；
- Go/etcd；
- NVLink/MNNVL；
- CUDA/PyTorch ABI、store/transfer-engine 构建选项和缓存键。

Mooncake Store 只能使用 JD 内部镜像仓库。内部包缺失或无法解析时立即停止并报告，禁止切换公共源规避。

### 决策输出

每个组件输出：

- `no change`，附证据；或
- 必须修改的依赖、构建选项、缓存失效和 clean rebuild 计划；或
- 阻塞项和需要用户批准的选择。

所有 prerequisite、build-option、dependency、architecture 改动都必须进入 single final JD CI commit，旧分支不可修改。

## 3. 生成只读计划

```bash
python3 .agents/skills/xq-sglang-jd-release-rebase/scripts/jd_release_rebase.py plan \
  --repo "$PWD" \
  --old-internal refs/remotes/origin/JD-v0.5.14 \
  --old-upstream refs/tags/v0.5.14 \
  --new-upstream refs/tags/v0.5.15 \
  --new-internal JD-v0.5.15 \
  --output /tmp/jd-release-rebase-plan.json
```

helper 使用拓扑顺序、`git cherry` 和 patch 等价性，输出：

- `replay_commits`：仍属 JD production 的 patch；
- `deferred_ci_commits`：触及 JD CI boundary、等待最终单一 commit 的历史；
- `absorbed_commits`：已被上游等价 patch 吸收；
- `audit_merge_commits`：不直接重放但必须审查的 merge commit；
- `high_risk_paths`：依赖、模型、分布式、SGL-Kernel 和 CI 高风险路径。

逐项核对 commit 顺序、文件和高风险原因。

## 4. 语义等价审查

对每个 `replay_commits` 搜索新上游实现、调用点和测试：

```bash
git show <old-jd-sha>
git log --oneline refs/tags/<old>..refs/tags/<new> -- <path>
git diff refs/tags/<old>..refs/tags/<new> -- <path>
rg -n "<symbol-or-behavior>" <new-upstream-worktree>
```

若外部修复确实覆盖相同问题与最终行为，在用户确认后运行：

```bash
python3 .agents/skills/xq-sglang-jd-release-rebase/scripts/jd_release_rebase.py classify \
  --plan /tmp/jd-release-rebase-plan.json \
  --absorbed <old-jd-sha> \
  --reason "upstream <sha> provides equivalent behavior"
```

证据不足时不调用 `classify`。

## 5. 重放 production commit

```bash
python3 .agents/skills/xq-sglang-jd-release-rebase/scripts/jd_release_rebase.py execute \
  --plan /tmp/jd-release-rebase-plan.json \
  --worktree ../sglang-JD-v0.5.15 \
  --state-file .git/jd-release-rebase/JD-v0.5.15.json
```

helper 在新上游 SHA 创建 isolated worktree，逐个重放 production commit，记录 old-to-new SHA。返回码 `3` 表示冲突已安全写入状态文件。

冲突处理：

1. 读取状态文件的 `pending_commit`、`conflict_files` 和映射。
2. 分别查看 upstream、新 JD patch 和目标文件调用路径。
3. 逐 hunk 重构最终行为，不能整文件选择一侧。
4. 不得顺手改变数值容差、性能阈值、依赖版本、硬件范围或模型语义。
5. 解决并加入 Git index 后运行：

```bash
python3 .agents/skills/xq-sglang-jd-release-rebase/scripts/jd_release_rebase.py resume \
  --state-file .git/jd-release-rebase/JD-v0.5.15.json
```

不得绕开状态文件手工创建 replay commit。

## 6. 迁移 JD CI 映射

应用全部延后 JD CI 历史，但不创建中间 commit：

```bash
python3 .agents/skills/xq-sglang-jd-release-rebase/scripts/jd_release_rebase.py prepare-ci \
  --state-file .git/jd-release-rebase/JD-v0.5.15.json
```

若返回码为 `3`，逐 hunk 解决、加入 Git index，然后运行 `resume-ci`。

更新 `test/jd-ci/jd_test_manifest.py`：

- replayed old SHA 替换为 new SHA；
- 已被上游吸收的 case 退役；
- 保留仍验证 JD 专有逻辑的断言；
- `operator_correctness` 与 `operator_performance` 成对；
- 删除 deferred old CI SHA；
- 最终 CI commit 自身引入的 case 使用 `tracks_ci_head=true`，不得嵌入自身 SHA。

调用 `$xq-sglang-jd-new-pr` 审查兼容改动和缺失测试。manifest 仍执行固定累积全量清单。

将 manifest、依赖适配、构建兼容、文档、测试和两个 Skill 全部加入 Git index 后，创建唯一 JD CI commit：

```bash
python3 .agents/skills/xq-sglang-jd-release-rebase/scripts/jd_release_rebase.py commit-ci \
  --state-file .git/jd-release-rebase/JD-v0.5.15.json \
  --message "ci: migrate JD CI for v0.5.15"
```

helper 会把所有 deferred old CI SHA 映射到该 commit，并拒绝中间插入的 commit。该 commit 在验证和发布期间始终必须是 `HEAD`。

## 7. 检查与本地验证

```bash
python3 .agents/skills/xq-sglang-jd-release-rebase/scripts/jd_release_rebase.py check \
  --state-file .git/jd-release-rebase/JD-v0.5.15.json \
  --output /tmp/jd-release-rebase-check.json
```

要求：

- 新 HEAD 继承新上游 tag；
- 所有 commit 已映射或明确跳过；
- conflict marker 为空；
- `git range-diff` 符合预期；
- manifest 没有 stale old SHA 或 missing new SHA；
- `jd_ci_commit_count=1`、`jd_ci_commit_is_head=true`；
- worktree clean。

本地执行：

```bash
python3 test/jd-ci/jd_test_manifest.py \
  --source "$PWD" --output /tmp/jd-cases.json
python3 -m unittest discover -s test/jd-ci/unit/ci -p 'test_*.py' -v
bash -n test/jd-ci/run_jd_ci.sh test/jd-ci/env/*.sh test/jd-ci/pipeline/*.sh
JD_CI_SERVER_API_DRY_RUN=1 \
  bash test/jd-ci/pipeline/run_server_api_regression.sh "$PWD" /tmp/jd-ci
JD_CI_OPERATOR_DRY_RUN=1 JD_CI_OPERATOR_AVAILABLE_GPUS=8 \
  bash test/jd-ci/pipeline/run_operator_regression.sh "$PWD" /tmp/jd-ci
```

必须复核以下固定设计：

- 恰好一个 `Qwen2.5-VL-7B-Instruct/` dummy-weight Server；
- `SERVER_CASES` 中的 15 个固定 HTTP 子 case；
- `--disable-cuda-graph`；
- `token_oracle` 与 KV canary；
- 上游 mock-model 和模型特定 deterministic CPU fixtures；
- `case_progress.py` 每 5 秒心跳；
- 执行全部固定 JD case；
- 普通 case 失败后继续。

## 8. 流水线机器真实验证

本地检查完成后，必须到用户指定的流水线机器执行真实验证；没有远端证据不能宣称升级完成。

- 使用独立源码目录和 tmux 会话，记录远端 commit、CUDA、GPU、基础镜像、依赖缓存和完整命令。
- 先预取并核验目标版本 `_deps` 的精确 revision。
- prerequisite、build-option、dependency 或 architecture 发生变化时，必须使旧 wheel cache 失效并 clean rebuild。
- 运行固定全部 JD case，确认单 Server、CPU/mock、算子 correctness/performance、5 秒心跳和失败汇总。
- 除非用户明确授权，验证不得保存、发布或推送任何镜像。用户要求“不产出镜像”时只能使用不会进入镜像保存逻辑的验证方式。
- `-m` 允许任意分支只读复用对应版本主分支的正式缓存，固定跳过测试并快速产出
  当前 commit 镜像；任一 cache miss 都立即失败，不得回退源码编译。镜像发布仍需用户单独授权。
- Mooncake Store 内部仓库失败时立即报告。

## 9. 发布

只有获得单独授权后才可 push。推送前重新运行 `check`，确认唯一 JD CI commit 仍为 HEAD。若目标分支已发布且必须改写，只能在用户明确确认后使用 `--force-with-lease`。

## 10. 结果报告

报告：

1. 四个 refs/SHAs；
2. replayed、absorbed、skipped、deferred commit；
3. 完整 old-to-new SHA 映射；
4. 冲突及处理依据；
5. SGL-Kernel/Mooncake prerequisite 决策和精确 revision；
6. 唯一 JD CI HEAD commit；
7. 本地与流水线机器命令、环境、结果和日志路径；
8. 未验证项、阻塞和镜像状态；
9. 进化报告。

不得把“未执行”表述为“通过”，不得把 dummy/mock 结果表述为真实模型精度保证。
