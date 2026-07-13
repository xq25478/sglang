# JD CI 执行模式与临时验证镜像设计

状态：已落地。

## 目标

- 保持正常 PR 的正式评审与合入行为清晰、严格。
- 为临时分支提供独立的验证镜像模式。
- 临时分支按用户确认结果决定是否编译 SGL-Kernel 和 Mooncake。
- 临时编译产物不得读取、覆盖或污染正式缓存。
- 在脚本入口提供完整、可直接操作的帮助说明。

## 非目标

- 不改变 Mooncake、SGL-Kernel 的主体编译命令。
- 不改变 CPU/Mock、Server/API、算子正确性与性能三类固定回归内容。
- 不根据 commit diff 自动判断组件是否需要编译。
- 不在 `test/jd-ci/` 之外增加 JD CI 文件。

## 命令行接口

`test/jd-ci/run_jd_ci.sh` 支持三个互斥模式：

| 参数 | 模式 | 组件产物 | JD CI 测试 | 镜像 |
| --- | --- | --- | --- | --- |
| `-r`, `--review` | 正常 PR 评审 | 编译并更新正式缓存 | 固定全量执行 | 不产出 |
| `-m`, `--merge` | 主分支合入 | 只允许安装正式缓存 | 固定跳过 | 产出正式镜像 |
| `-t`, `--temp-image` | 临时分支验证 | 继承基础镜像或在临时目录编译 | 由 `JD_CI_SKIP_TEST` 控制 | 产出临时验证镜像 |

同时支持：

- `-h`, `--help`：打印完整帮助并以状态码 0 退出；
- 无参数：保持兼容，等价于 `-r`；
- `note__merge_request`：保持兼容，等价于 `-r`；
- `merge_request__merged`：保持兼容，等价于 `-m`；
- 未知参数：打印错误和帮助并以状态码 2 退出。

## 正式评审与合入

### `-r` 正常 PR 评审

- 使用 `JD-${BASE_IMAGE_TAG}` 对应的正式缓存目录。
- 强制编译 SGL-Kernel、Mooncake TE 和 Mooncake-store。
- 编译完成后安装产物并更新正式缓存。
- 固定执行 CPU/Mock、Server/API、算子正确性与性能三类全部回归。
- 任一回归失败不阻止其他回归执行，最终统一返回失败。
- 无论测试结果如何都不创建或推送镜像。

### `-m` 主分支合入

- 仅允许在当前分支等于 `JD-${BASE_IMAGE_TAG}` 时运行。
- 使用与 `-r` 相同的正式缓存目录。
- SGL-Kernel、Mooncake TE 或 Mooncake-store 缓存缺失时直接失败，不允许源码编译兜底。
- 固定跳过 JD CI 测试，因为测试已由对应的 `-r` 完成。
- 所有正式缓存安装成功后，创建并推送正式 SGLang 和 Mooncake-store 镜像。

`JD_CI_SKIP_SGL_KERNEL_BUILD`、`JD_CI_SKIP_MOONCAKE_BUILD` 和
`JD_CI_SKIP_TEST` 属于临时镜像选项；在 `-r` 或 `-m` 中设置为 `1` 时应直接报错，
避免削弱正式流程。

## `-t` 临时验证镜像

### 组件确认

| 环境变量 | 默认值 | 值为 `0` | 值为 `1` |
| --- | --- | --- | --- |
| `JD_CI_SKIP_SGL_KERNEL_BUILD` | `0` | 临时编译并安装 SGL-Kernel | 用户确认无相关改动，继承基础镜像版本 |
| `JD_CI_SKIP_MOONCAKE_BUILD` | `0` | 临时编译并安装 Mooncake TE 与 Mooncake-store | 用户确认无相关改动，继承基础镜像版本 |
| `JD_CI_SKIP_TEST` | `0` | 固定执行全部 JD CI 测试 | 跳过全部 JD CI 测试 |

三个变量只接受 `0` 或 `1`。非法值必须在 tag 查询、目录创建和 Docker 操作之前失败。

组件 SKIP 由用户根据代码改动显式确认，流水线不做 diff 推断：

- SKIP 为 `0`：使用 `${CI_ARTIFACT_ROOT}/tmp_artifacts/${COMMIT_SHA}/...`
  下的组件独立目录，强制源码编译并安装；
- SKIP 为 `1`：不调用对应构建脚本，不挂载正式 wheel 作为安装来源，直接继承基础镜像中的组件；
- 临时目录在成功、失败、中断和镜像推送完成后都由统一清理逻辑删除。

### 测试与镜像

- `JD_CI_SKIP_TEST=0`：执行全部三类 JD CI 回归；全部回归通过后才产出镜像；
- `JD_CI_SKIP_TEST=1`：跳过三类 JD CI 回归，组件处理成功后直接产出镜像；
- `JD_CI_SKIP_TEST=1` 必须在日志和汇总中明确记录测试被用户显式跳过；
- 临时模式把 SGLang 和 Mooncake-store 两张验证镜像视为同一组产物；
- 任一组件处理、容器执行或测试失败时，两张临时镜像都不得推送，避免留下不完整的验证组合；
- 全部门禁通过后再依次创建并推送 SGLang 和 Mooncake-store 两张验证镜像；
- 临时镜像使用包含分支名和 commit 的独立 tag，避免覆盖正式镜像和其他临时分支镜像。

建议 tag：

```text
SGLang:         ${BASE_IMAGE_TAG}_JD_TMP_${BRANCH_NAME_FOR_DOCKER}_${COMMIT_ID}
Mooncake-store: ${MSTORE_IMAGE_TAG}_JD_TMP_${BRANCH_NAME_FOR_DOCKER}_${COMMIT_ID}
```

tag 继续执行现有的小写化处理；分支名中的 `/` 和非 `[a-z0-9_.-]` 字符统一替换为 `-`，
连续分隔符压缩为一个，并按 Docker tag 长度限制截断。

## 帮助信息

帮助必须覆盖：

- `-r`、`-m`、`-t`、`-h` 的含义；
- 三种模式的编译、缓存、测试和镜像行为；
- 三个临时镜像环境变量的默认值和 `0/1` 行为；
- 正式缓存与临时目录的隔离原则；
- 缓存缺失、测试失败和非法参数的失败规则；
- 正常评审、主分支合入、临时全编译、按组件跳过、跳过测试等示例。

帮助函数必须位于脚本开头；`-h` 不得触发 Git、Docker、目录创建或任何外部副作用。

目标帮助文案：

```text
用法:
  test/jd-ci/run_jd_ci.sh [-r | -m | -t | -h]

执行模式（互斥）:
  -r, --review       正常 PR 评审；默认模式。
                     强制编译 SGL-Kernel 和 Mooncake，更新正式缓存，
                     固定执行全部 JD CI 回归，不产出镜像。

  -m, --merge        正式分支合入。
                     只安装 -r 生成的正式缓存，缓存缺失立即失败，
                     不执行 JD CI 回归，产出正式 SGLang 和 Mooncake-store 镜像。

  -t, --temp-image   临时分支验证镜像。
                     组件默认在 commit 独立临时目录中编译并安装；
                     可由用户显式跳过组件编译或全部测试；
                     所有已启用门禁通过后产出两张临时验证镜像并清理临时目录。

  -h, --help         显示本帮助并退出。

临时镜像选项（仅用于 -t，只接受 0 或 1）:
  JD_CI_SKIP_SGL_KERNEL_BUILD  默认 0；1 表示继承基础镜像中的 SGL-Kernel。
  JD_CI_SKIP_MOONCAKE_BUILD    默认 0；1 表示继承基础镜像中的 Mooncake。
  JD_CI_SKIP_TEST              默认 0；1 表示跳过全部 JD CI 回归。

约束:
  * 无参数等价于 -r。
  * -r 和 -m 不允许通过上述变量跳过正式流程。
  * -m 只能在 JD-${BASE_IMAGE_TAG} 正式分支运行。
  * -t 不能在正式分支运行，且不会读取、写入或覆盖正式组件缓存。
  * -t 的任一组件、容器或测试失败时，不推送两张临时镜像。

示例:
  test/jd-ci/run_jd_ci.sh -r
  test/jd-ci/run_jd_ci.sh -m
  test/jd-ci/run_jd_ci.sh -t
  JD_CI_SKIP_SGL_KERNEL_BUILD=1 test/jd-ci/run_jd_ci.sh -t
  JD_CI_SKIP_MOONCAKE_BUILD=1 test/jd-ci/run_jd_ci.sh -t
  JD_CI_SKIP_SGL_KERNEL_BUILD=1 JD_CI_SKIP_MOONCAKE_BUILD=1 \
    test/jd-ci/run_jd_ci.sh -t
  JD_CI_SKIP_TEST=1 test/jd-ci/run_jd_ci.sh -t
```

## 运行决策

脚本先完成纯参数处理，再进入现有容器主体：

1. 归一化命令行参数为 `review`、`merge` 或 `temp-image`。
2. 校验临时镜像环境变量只能为 `0/1`。
3. 校验 `merge` 只能在正式分支运行，`temp-image` 不能在正式分支运行。
4. 按模式选择正式缓存或 commit 临时目录。
5. 按模式和组件 SKIP 决定编译、缓存安装或继承基础镜像。
6. 按模式和 `JD_CI_SKIP_TEST` 决定是否执行固定全量回归。
7. 按模式与执行结果决定不产出、产出正式镜像或产出临时镜像。
8. 始终执行已有容器、运行时临时目录和临时组件产物清理。

## 失败处理

- 参数或环境变量非法：状态码 2，打印帮助，不进入 Docker。
- `-m` 不在正式分支：状态码 2，不产出镜像。
- `-t` 在正式分支：状态码 2，提示使用 `-r` 或 `-m`。
- `-m` 正式缓存缺失：流水线失败，不允许源码编译兜底。
- `-t` 任一组件编译或容器执行失败：两张临时镜像都不推送，清理临时目录。
- `-t` 且 `JD_CI_SKIP_TEST=0` 时任一测试失败：两张临时镜像都不推送。
- 所有失败继续使用现有日志转储和退出码汇总机制。

## 验证要求

- 为参数别名、默认 `-r`、`-h` 和未知参数增加 shell 行为契约。
- 为三种模式的缓存作用域、构建决策、测试决策和镜像决策增加契约测试。
- 为三个 `0/1` 环境变量及非法值增加契约测试。
- 保留 `-r` 全部回归不可跳过的现有契约。
- 验证 `-m` 的 SGL-Kernel 与 Mooncake 缓存 miss 都会失败。
- 验证 `-t` 的临时目录包含完整 commit SHA，并在退出时清理。
- 验证临时镜像 tag 不等于正式镜像 tag，且不同分支不会互相覆盖。
- 运行完整 `test/jd-ci/unit/ci`、全部现存 shell 语法检查和三类回归 dry-run。
- 未得到用户明确授权时，不在验证过程中创建或推送任何镜像。
