# SPECTRE 使用说明

## 1. 概述

`SPECTRE` 是 SGLang 中的远程 speculative decoding 模式。

它把 speculative decoding 拆成两个服务：

- `Target server`：加载大模型并对外提供服务。
- `Draft server`：加载小模型并为 target server 生成远程 draft tokens。

与本地 speculative decoding 相比，SPECTRE 会把 draft 生成移动到另一个进程或另一台机器上。它适用于以下场景：

- 大模型开销较高，希望主要聚焦在 verify 上；
- 小模型可以部署在独立 GPU 或独立机器上；
- 希望研究 target / draft 解耦、网络传输开销以及端到端 speculative 行为。

## 2. 架构

基础数据流如下：

1. 客户端只向 `Target server` 发送请求。
2. `Target server` 通过 ZMQ 把 draft 请求发给 `Draft server`。
3. `Draft server` 运行小模型并返回 draft tokens。
4. `Target server` 使用大模型对这些 draft tokens 做 verify。
5. `Target server` 将最终结果返回给客户端。

当前实现中，`SPECTRE` 使用：

- 客户端与 target 之间的常规 HTTP / OpenAI 兼容接口；
- target 与 draft 之间的 ZMQ 通信；
- `REMOTE` 作为 speculative algorithm；
- `target` / `draft` 作为运行角色。

## 3. 环境配置

### 3.1 前置要求

当前 SPECTRE 实现基于 `SGLang 0.5.10rc` 开发。

推荐环境：

- Linux
- Python `>= 3.10`
- CUDA-enabled PyTorch
- 至少 2 张 GPU，或 2 个能够通过 ZMQ 通信的服务实例

如果 target 和 draft 部署在不同机器上，请确保：

- target 机器可以访问 draft 机器配置的 ZMQ 端口；
- 防火墙或安全组放通相关端口；
- 两台机器都能访问各自模型路径。

### 3.2 基础镜像 / 运行时

建议先准备一个 `sglang 0.5.10rc` 的环境镜像，再在其基础上应用当前 SPECTRE 代码。

推荐做法：

- 如果团队内部已经有 `0.5.10rc` 镜像，优先直接使用；
- 否则可以先从公开 SGLang 镜像仓库选择接近版本的镜像，再同步当前仓库代码。

### 3.3 构建 C++ ZMQ 组件

SPECTRE 依赖下面这个目录中的 C++ ZMQ 扩展：

`python/sglang/srt/speculative/remote_spec/cpp_zmq/`

可以直接参考：

`python/sglang/srt/speculative/remote_spec/cpp_zmq/build_cpp_zmq.sh`

等价命令如下：

```bash
cd python/sglang/srt/speculative/remote_spec/cpp_zmq

pip config set global.index-url https://mirrors.jd.com/pypi/web/simple
pip config set global.trusted-host mirrors.jd.com
pip install pybind11 msgpack

apt-get update && apt-get install -y libmsgpack-dev

python3 setup.py build_ext --inplace
```

如果你的环境不方便修改全局 pip 配置，可以仅使用你自己的源安装 Python 依赖。

### 3.4 先验证基础服务

在尝试 SPECTRE 之前，建议先确认普通 SGLang 服务可以正常启动：

```bash
python -m sglang.launch_server \
  --model-path /path/to/model \
  --host 0.0.0.0 \
  --port 30000
```

如果普通服务无法正常启动，请先修复基础环境问题，再继续配置 SPECTRE。

## 4. 核心概念

### 4.1 Target server

Target server 负责：

- 加载大模型；
- 接收用户请求；
- 发送远程 draft 请求；
- 校验远程 draft tokens；
- 返回最终生成结果。

你的 benchmark 脚本或在线请求应当只访问 target server。

### 4.2 Draft server

Draft server 负责：

- 加载小模型；
- 接收来自 target server 的远程 draft 请求；
- 维护 draft 侧请求状态；
- 把 draft tokens 返回给 target server。

Draft server 不应该作为普通对外服务直接暴露给用户。

### 4.3 大模型与小模型

一个典型部署中：

- `big model` 对应 `Target server`；
- `small model` 对应 `Draft server`。

例如：

- target：`Qwen3-32B`
- draft：`Qwen3-0.6B` 或 `Qwen3-1.5B`

## 5. 如何启动 SPECTRE

以下是实际可参考的启动示例。请根据你的模型路径、TP 大小、端口和机器地址自行替换。

### 5.1 示例

#### 启动 draft server（小模型）

```bash
python -m sglang.launch_server \
  --model-path /path/to/small-model \
  --tp 1 \
  --served-model-name small-model-name \
  --mem-fraction-static 0.8 \
  --decode-log-interval 1 \
  --host 0.0.0.0 \
  --port 3008 \
  --max-running-requests 256 \
  --chunked-prefill-size 16384 \
  --watchdog-timeout 60000 \
  --dist-timeout 6000000 \
  --context-length 16384 \
  --trust-remote-code \
  --cuda-graph-max-bs 128 \
  --disable-overlap-schedule \
  --remote-speculative-role draft \
  --remote-speculative-zmq-port 30090 \
  --remote-speculative-draft-priority \
  --remote-speculative-max-draft-priority-steps 10 \
  --log-level info \
  --remote-speculative-max-batch-size 3200
```

#### 启动 target server（大模型）

```bash
python -m sglang.launch_server \
  --model-path /path/to/big-model \
  --tp 1 \
    --served-model-name big-model-name \
    --mem-fraction-static 0.8 \
    --decode-log-interval 1 \
    --host 0.0.0.0 \
    --port 30000 \
    --max-running-requests 256 \
    --chunked-prefill-size 16384 \
    --watchdog-timeout 60000 \
    --dist-timeout 6000000 \
    --context-length 16384 \
    --trust-remote-code \
    --cuda-graph-max-bs 128 \
    --disable-overlap-schedule \
    --speculative-algorithm REMOTE \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --remote-speculative-role target \
    --remote-speculative-max-batch-size 1600 \
    --remote-speculative-reject-interval 2000 \
    --remote-speculative-retry-fail-ratio 0.6 \
    --remote-speculative-retry-min-count 15 \
    --remote-speculative-zmq-addr 127.0.0.1 \
    --remote-speculative-zmq-port 30090 \
    --log-level info \
    --attention-backend fa3
```

两个服务都启动完成后，客户端只需要访问：

```bash
http://127.0.0.1:30000
```

## 6. 参数说明

启用 SPECTRE 时，以下参数最关键：

| 参数 | 说明 | 默认值 |
|--------------------|--------------------|--------------------|
| `--speculative-algorithm` | Speculative algorithm。SPECTRE / RemoteSpec 需要设置为 `REMOTE`。 | `None` |
| `--remote-speculative-role` | 当前服务角色。`target` 表示大模型 verify 侧，`draft` 表示小模型 draft 侧。 | `None` |
| `--remote-speculative-max-batch-size` | 服务被认为进入高开销状态的阈值。当 running batch size 超过该值时，draft 可能停止处理 draft 请求，target 也可能停止发送。 | `32` |
| `--remote-speculative-reject-interval` | draft 侧 reject 或 draft 帮助不足时，target 重新发送 draft 请求的间隔。`1` 表示每轮都重发。 | `5000` |
| `--remote-speculative-no-draft-ratio` | 没有可用 draft tokens 的请求占总 batch 的比例。如果超过该值，target 会退化成一次只 decode 一个 token。 | `0.5` |
| `--remote-speculative-retry-fail-ratio` | 触发 retry 或 fallback 行为所需的失败 pre-verify 请求占 batch 的最小比例。 | `0.5` |
| `--remote-speculative-retry-min-count` | 触发 retry 或 fallback 行为所需的最小失败请求数。 | `4` |
| `--remote-speculative-zmq-addr` | SPECTRE / RemoteSpec 使用的 ZMQ 地址。同机部署通常为 `127.0.0.1`；跨机部署应填写 draft 所在机器 IP。 | `None` |
| `--remote-speculative-zmq-port` | target 与 draft 之间使用的 ZMQ 端口。两边必须一致。 | `None` |
| `--remote-speculative-draft-priority` | 是否开启 draft 侧对远程 draft 请求的优先调度。 | `False` |
| `--remote-speculative-max-draft-priority-steps` | 开启 draft priority 后，draft 侧最多连续优先调度的步数。 | `0` |
| `--speculative-num-steps` | 一轮 speculative 中采样的 step 数，会影响 draft tree 深度和 target verify 深度。 | `None` |
| `--speculative-eagle-topk` | 每个 speculative step 的分支宽度。虽然 SPECTRE 不是本地 EAGLE，但仍复用了树形 verify 结构，因此该参数仍然控制 tree width。 | `None` |
| `--speculative-num-draft-tokens` | 一轮 speculative 中 verify 的 draft token 数。一个常见的单路径配置是 `steps=3`、`topk=1`、`draft_tokens=4`。 | `None` |
| `REMOTE_SPEC_RECV_TIMEOUT_MS` | target 等待 draft 响应的最长时间，单位毫秒。该项通过环境变量控制，不是 CLI 参数。 | `200` |
| `REMOTE_SPEC_FAILURE_THRESHOLD` | 连续多少轮未收到有效 draft 响应后，target 暂时停止发送 draft 请求。该项通过环境变量控制，不是 CLI 参数。 | `30` |
| `REMOTE_SPEC_COOLDOWN_ROUNDS` | target 进入 cooldown 后，经过多少轮再恢复发送 draft 请求。该项通过环境变量控制，不是 CLI 参数。 | `100` |
| `REMOTE_SPEC_DEBUG` | 是否开启 ZMQ 相关调试日志。该项通过环境变量控制，不是 CLI 参数。 | `0` |

如果你想先从最简单的配置开始，推荐：

```text
speculative-num-steps = 3
speculative-eagle-topk = 1
speculative-num-draft-tokens = 4
```

## 7. 冒烟测试

启动完成后，先对 target 做健康检查：

```bash
curl http://127.0.0.1:30000/health
```

然后发送一个简单的生成请求：

```bash
curl -X POST http://127.0.0.1:30000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Write one sentence about speculative decoding.",
    "sampling_params": {
      "temperature": 0.0,
      "max_new_tokens": 32
    }
  }'
```

如果请求成功，说明 target 侧基础服务是通的。  
如果没有看到明显 speculative 收益，优先检查：

- target 日志；
- draft 日志；
- ZMQ 地址和端口；
- 两边是否都按预期启用了 `REMOTE` 相关参数；
- 是否某一侧经常进入 high-overhead fallback。

## 8. Benchmark

仓库中已经包含一些 benchmark 辅助脚本，例如：

- `SPECTRE-pressure/start_pressure.sh`

典型流程：

1. 启动 draft server。
2. 启动 target server。
3. benchmark 脚本对 target server 的 HTTP 接口发压测请求。

## 9. 结果图

最终结果图加入仓库后，请把下面的占位符替换成真实图片路径：

```markdown
![SPECTRE Result](./images/remote_spec_result_placeholder.png)
```

如果你更希望保留显式 TODO，也可以用：

```text
[TODO] 在这里插入最终的 SPECTRE 结果图
```
