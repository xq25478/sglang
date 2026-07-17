# JD CI 单镜像流程实现记录

状态：已完成。

## 已实现约束

- 参数解析和帮助输出先于 Git、Docker 和目录操作。
- `-r` 强制编译 SGL-Kernel、Mooncake TE，并完整执行全部 JD case。
- `-m` 只读复用正式缓存，任一 cache miss 立即失败，成功后发布一张 SGLang 镜像。
- `-t` 使用 commit runner 隔离编译产物，默认执行全部 JD case，成功后发布一张
  SGLang 镜像。
- 主镜像内的 Mooncake TE 编译命令保持不变。
- 独立第二容器、第二组件缓存、第二份容器日志、第二次镜像提交和推送均已删除。
- 所有模式无论成功、失败还是中断都清理主容器、runner 和 final-state 目录。

## 契约

`test/jd-ci/unit/ci/test_internal_ci_contract.py` 和
`test/jd-ci/unit/ci/test_cpu_mock_regression_runner.py` 固定检查：

- 三种模式及两个历史事件名行为；
- 正式缓存与临时目录隔离；
- 主容器中 SGL-Kernel、Mooncake TE 构建调用各一次；
- 三类 JD 回归各执行一次；
- 发布路径只有一次 `docker commit` 和一次 `docker push`；
- 失败摘要在现场清理后仍输出主流水线根因。

## 验证命令

```bash
bash -n test/jd-ci/run_jd_ci.sh test/jd-ci/env/*.sh test/jd-ci/pipeline/*.sh
python3 -m unittest discover -s test/jd-ci/unit/ci -p 'test_*.py' -v
git diff --check
```
