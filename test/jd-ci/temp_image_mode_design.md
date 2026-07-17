# JD CI 执行模式与临时验证镜像设计

状态：已落地。

## 目标

- `-r` 为正常 PR 执行完整组件编译和全部 JD 回归，不发布镜像。
- `-m` 为任意分支只读复用正式缓存，快速发布一张 SGLang 镜像。
- `-t` 为临时分支使用 commit 隔离目录编译、测试并发布一张 SGLang 镜像。
- 兼容 `note__merge_request` 和 `merge_request__merged` 两个历史事件名。

## 单镜像边界

CI 只启动主 SGLang 容器，只创建并推送 SGLang 镜像。SGL-Kernel 和 Mooncake TE
仍在主容器中按原有编译脚本构建、安装和缓存。CI 不再启动独立的 Mooncake 容器，
不再编译独立 store 产物，也不再创建、提交、推送或清理第二张镜像。

## 模式

| 参数 | 组件产物 | JD 回归 | 镜像 |
| --- | --- | --- | --- |
| `-r`, `--review` | 强制编译并更新正式缓存 | 固定全量执行 | 不产出 |
| `-m`, `--merge` | 只安装正式缓存，cache miss 失败 | 固定跳过 | 一张 SGLang 镜像 |
| `-t`, `--temp-image` | 继承基础镜像或在临时目录编译 | 默认全量，可显式全部跳过 | 一张 SGLang 镜像 |

`-t` 的 `JD_CI_SKIP_SGL_KERNEL_BUILD`、`JD_CI_SKIP_MOONCAKE_BUILD` 和
`JD_CI_SKIP_TEST` 只接受 `0` 或 `1`。正式模式不接受跳过选项。临时编译产物只写入
本次 runner 目录，不读取或覆盖正式缓存。

## 发布门禁与清理

只有主容器内已启用的组件安装和测试全部成功时，`-m` 或 `-t` 才执行一次
`docker commit` 和一次 `docker push`。失败时先转储日志和提取根因，再清理主容器、
runner 目录与 final-state 缓冲；成功、中断时也执行同一套清理。

镜像标签统一为：

```text
images-infra-cn-east-1-inner.jcr.service.jdcloud.com/sglang:${BASE_IMAGE_TAG}_JD_${COMMIT_ID}
```

## 验证要求

- Shell 语法检查通过。
- `test/jd-ci/unit/ci` 全部契约通过。
- 契约明确断言只存在一个容器发布路径、一次 `docker commit` 和一次 `docker push`。
- 未得到用户明确授权时，验证不得创建或推送镜像。
