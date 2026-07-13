# JD CI 单模型完整功能回归与实时日志设计

## 目标

JD CI 每次评审执行固定、累积的 JD 内部回归，但只启动一个 SGLang Server。
测试日志必须让使用者随时知道当前正在执行哪个 case、验证什么、已经运行多久、
距离超时还有多长时间，以及 case 是否通过。

本设计保持 Mooncake、SGL-Kernel、容器、缓存、镜像和清理主体结构不变，只调整
`test/jd-ci` 内的测试清单、Server/API 回归和三类回归的实时日志。

## 执行模式

- `-r`：启动唯一 Server，执行全部 CPU/Mock、Server/API、算子正确性与性能 case，
  不创建镜像。
- `-m`：不执行测试，只消费正式缓存并发布正式镜像。
- `-t`：`JD_CI_SKIP_TEST=0` 时执行与 `-r` 相同的全部 case；显式设置为 `1` 时
  记录全部测试被用户跳过。
- 测试清单固定累积，不根据当前 diff 动态选择。

## 唯一 Server

Server 使用以下配置：

- 模型目录：`/mnt/nas/models/Qwen2.5-VL-7B-Instruct/`
- 加载方式：`--load-format dummy`
- 一个可见 GPU，`--tp-size 1`
- 使用确定性的采样配置，避免生成内容随机性影响功能断言

整个 Server/API 回归只启动一次该 Server。所有 HTTP case 共享进程，结束后统一
清理。CI 不启动 GLM、Kimi、DeepSeek 或 Qwen3.5 checkpoint。

## 覆盖清单

### 唯一 Server 上的端到端 case

1. 文本非流式请求成功，响应结构完整。
2. 文本流式请求成功，至少返回一个有效 chunk 并正常结束。
3. 图像非流式请求成功，覆盖多模态输入预处理和 HTTP 路径。
4. 图像流式请求成功，覆盖多模态流式响应。
5. 损坏的 base64 图片在非流式请求中返回 HTTP 400，Server 保持存活。
6. 无效图片 URL 在非流式请求中返回 HTTP 400，Server 保持存活。
7. 损坏的 base64 图片在流式请求中不能返回 HTTP 500，Server 保持存活。
8. 无效图片 URL 在流式请求中不能返回 HTTP 500，Server 保持存活。
9. `ignore_eos` 开启时生成达到配置的 token 数，而不是只检查环境开关。
10. `thinking` 为 list、缺少 `type` 的 dict、未知字符串和整数时，请求被安全归一化
    或忽略，不能造成 HTTP 500 或 Server 退出。

### 不启动专属模型的确定性 case

- GLM reasoning parser：使用固定的模拟模型输出，分别覆盖 reasoning enabled、
  disabled、非法输入和流式增量解析，断言最终 `content`、`reasoning_content` 与
  `reasoning_tokens`。
- Kimi K2 function call：使用固定协议文本覆盖被引号包裹的 arguments object。
- DSV4 function call：使用标量、数组和非法 JSON arguments，验证编码过程不拒绝
  JD 支持的输入。
- invalid `thinking`：四种非法类型分别作为可见子 case，同时保留 HTTP 路径与
  CPU 协议归一化断言。
- invalid `thinking` 的归一化发生在 `ChatCompletionRequest` 校验层，早于 tokenizer、
  chat template 和模型推理；因此不加载 DeepSeek-V4-Flash 权重。另用 CPU 单测将
  `reasoning_parser` 固定为 `deepseek-v4`，覆盖 DeepSeek 专属 reasoning
  enable、disable 和默认关闭分支。
- 现有 JD runtime、metrics、cache、部署配置和四组算子 correctness/performance
  case 持续执行。

这些 case 验证模型专属 parser、协议和状态转换修复，但不宣称验证 GLM、Kimi、
DeepSeek 或 Qwen3.5 的真实 checkpoint 加载、生成质量或模型精度。

## 实时日志协议

三类 runner 使用相同的单行格式，并在开始时打印完整 case 清单。心跳周期固定为
5 秒，不提供关闭或修改周期的环境变量。

```text
[JD CI][Server and API Regression][CASE 2/15][START] id=jd-text-streaming assertion="文本流式响应完整" timeout=60s
[JD CI][Server and API Regression][CASE 2/15][RUNNING] id=jd-text-streaming phase=request elapsed=5s timeout=60s
[JD CI][Server and API Regression][CASE 2/15][PASS] id=jd-text-streaming duration=8s
```

唯一 Server 启动和关闭也作为可观察动作：

```text
[JD CI][Server and API Regression][SERVER][START] model=qwen25-vl load_format=dummy timeout=600s
[JD CI][Server and API Regression][SERVER][RUNNING] phase=startup elapsed=10s timeout=600s
[JD CI][Server and API Regression][SERVER][READY] duration=42s
```

日志字段要求：

- `CASE 当前序号/总数`
- 稳定 `case_id`
- 中文或清晰英文的断言目标
- 当前动作 `START`、`RUNNING`、`PASS`、`FAIL`、`BLOCKED` 或 `SKIP`
- `RUNNING` 中的 phase、elapsed 和 timeout
- 结束动作中的 duration；失败时同时给出 exit code、原因和 case 日志路径

命令原始输出继续实时写入流水线 stdout 和独立 case 日志，心跳不会替代原始输出。

## 失败与继续执行

- 普通 CPU/Mock、HTTP 或算子 case 失败后立即记录 `FAIL`，继续执行剩余固定 case。
- 唯一 Server 启动失败时，所有依赖 Server 的 case 记录为 `BLOCKED`；CPU/Mock 和
  算子 case 继续执行。
- HTTP case 失败但 Server 仍存活时继续下一个 HTTP case。
- Server 在执行中退出时，当前 case 记录为 `FAIL`，剩余 HTTP case 记录为
  `BLOCKED`，随后继续其他测试区域。
- GPU 数不足时，受影响 case 记录为 `BLOCKED`，报告必须说明需要和实际 GPU 数。
- 全部 case 完成后统一汇总；任一 `FAIL` 或未经允许的 `BLOCKED` 都使 CI 返回非零。

## 报告

每个 case 在 JSON 报告中保留：

- `name`、`status`、`assertion`
- `duration_seconds`、`timeout_seconds`
- `phase` 或最终执行动作
- `exit_code`、`detail`、`log_file`
- Server case 的模型配置和 HTTP 状态

Markdown 汇总列出测试区域总数、已完成数、通过、失败、阻塞、跳过和当前失败
case。stdout 的实时日志用于观察过程，JSON/Markdown 用于最终审计。

## 验证

- 契约测试固定 5 秒心跳、日志字段、完整清单和继续执行语义。
- CPU 环境使用 dry-run 验证所有 case 都被列出并写入报告。
- 使用短命令验证 `START -> RUNNING -> PASS/FAIL` 状态转换和信号清理。
- 流水线机器验证唯一 Qwen2.5-VL dummy Server 的全部 HTTP case。
- 完整 `-r` 验证三类回归均执行且没有镜像发布。

## 不在本次范围内

- GSM8K、MMLU 或其他模型精度评测。
- 多 checkpoint 启动与模型间输出对比。
- SGLang 原生 `test/registered`、`test/manual` 或完整 SGL-Kernel 原生测试套件。
- Mooncake、SGL-Kernel、缓存和镜像主体流程重构。
