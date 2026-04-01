# SPECTRE README

## 1. Overview

`SPECTRE` is the remote speculative decoding mode in SGLang.

It splits the speculative pipeline into two services:

- `Target server`: loads the large model and serves user traffic.
- `Draft server`: loads the small model and generates remote draft tokens for the target server.

Compared with local speculative decoding, SPECTRE moves draft generation to another process or another machine. This is useful when:

- the large model is expensive and should stay focused on verification;
- the small model can run on a separate GPU or machine;
- you want to study target/draft decoupling, network overhead, and end-to-end speculative behavior.

## 2. Architecture

The basic data flow is:

1. The client sends requests only to the `Target server`.
2. The `Target server` sends draft requests to the `Draft server` through ZMQ.
3. The `Draft server` runs the small model and returns draft tokens.
4. The `Target server` verifies those draft tokens with the large model.
5. The `Target server` returns the final result to the client.

In current code, `SPECTRE` uses:

- normal HTTP/OpenAI-compatible serving between client and target;
- ZMQ between target and draft;
- `REMOTE` as the speculative algorithm type;
- `target` / `draft` as the runtime role.

## 3. Environment Setup

### 3.1 Prerequisites

This SPECTRE implementation is developed based on `SGLang 0.5.10rc`.

Recommended environment:

- Linux
- Python `>= 3.10`
- CUDA-enabled PyTorch
- at least 2 GPUs or 2 services that can communicate through ZMQ

If you run target and draft on different machines, make sure:

- the target machine can reach the draft machine over the configured ZMQ port;
- firewalls and security groups allow that traffic;
- both machines can access their model paths.

### 3.2 Base image / runtime

We recommend preparing the environment from an `sglang 0.5.10rc` image first, then applying the SPECTRE code changes on top of it.

Recommended practice:

- if your team already has an internal `0.5.10rc` image, use that image directly;
- otherwise start from the closest available SGLang image in the Docker Hub repository above, then sync this repo and use the current SPECTRE code.

### 3.3 Build the C++ ZMQ component

SPECTRE also relies on the C++ ZMQ extension under:

`python/sglang/srt/speculative/remote_spec/cpp_zmq/`

You can build it by directly following:

`python/sglang/srt/speculative/remote_spec/cpp_zmq/build_cpp_zmq.sh`

Equivalent steps are:

```bash
cd python/sglang/srt/speculative/remote_spec/cpp_zmq

pip config set global.index-url https://mirrors.jd.com/pypi/web/simple
pip config set global.trusted-host mirrors.jd.com
pip install pybind11 msgpack

apt-get update && apt-get install -y libmsgpack-dev

python3 setup.py build_ext --inplace
```

If your environment does not allow changing global pip config, just install the required Python packages with your own index source.

### 3.4 Verify basic serving first

Before trying SPECTRE, it is recommended to verify that a normal server can start successfully:

```bash
python -m sglang.launch_server \
  --model-path /path/to/model \
  --host 0.0.0.0 \
  --port 30000
```

If this basic server cannot start, fix the environment first, then come back to SPECTRE.

## 4. Core Concepts

### 4.1 Target server

The target server:

- loads the large model;
- receives user requests;
- sends remote draft requests;
- verifies remote draft tokens;
- returns final generation results.

This is the only service your benchmark scripts or online clients should call.

### 4.2 Draft server

The draft server:

- loads the small model;
- receives remote draft requests from the target server;
- maintains draft-side request state;
- sends draft tokens back to the target server.

The draft server is not intended to be exposed to normal users.

### 4.3 Big model and small model

In a typical deployment:

- the `big model` is the `Target server` model;
- the `small model` is the `Draft server` model.

Example:

- target: `Qwen3-32B`
- draft: `Qwen3-0.6B` or `Qwen3-1.5B`

## 5. How To Start SPECTRE

Below are practical startup examples. Replace the model paths, TP size, ports, and host addresses with your own values.

### 5.1 Example

#### Start the draft server (small model)

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

#### Start the target server (big model)

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

After both are ready, send traffic only to:

```bash
http://127.0.0.1:30000
```


## 6. Parameter Guide

This section separates `SPECTRE-specific parameters` from the `general speculative parameters` that are also required in practice.

### 6.1 SPECTRE-specific parameters

#### `--remote-speculative-role`

Choices:

- `target`
- `draft`

Meaning:

- `target`: run the large-model verification side.
- `draft`: run the small-model draft-generation side.

This is the most important SPECTRE switch because it decides which scheduler path the server enters.

#### `--remote-speculative-max-batch-size`

Meaning:

- The high-overhead threshold for SPECTRE scheduling.
- If the current batch is larger than this value, the server may consider itself too expensive for SPECTRE behavior.

Typical use:

- lower it when you want the system to back off earlier under load;
- raise it when your hardware can still handle larger speculative batches well.

#### `--remote-speculative-reject-interval`

Meaning:

- Controls how often the target side resends draft requests after the draft side rejects or cannot effectively provide draft help.

Behavior summary:

- smaller value: retry more aggressively;
- larger value: retry less often;
- `1`: retry every time.

Typical use:

- reduce it for more aggressive draft re-engagement;
- increase it if repeated retries add overhead.

#### `--remote-speculative-no-draft-ratio`

Meaning:

- A target-side fallback threshold.
- If too many requests in the batch currently have no usable draft tokens, the target side falls back to decoding only one token.

Behavior summary:

- smaller value: fallback is triggered more easily;
- larger value: keep trying SPECTRE longer.

Typical use:

- use a smaller value when draft responses are unstable or often missing;
- use a larger value when draft service is stable and you want to preserve speculative behavior longer.

#### `--remote-speculative-retry-fail-ratio`

Meaning:

- A ordinary sepculative decode threshold.
- If too many requests in the batch currently rollback, the target side falls back to ordinary speculative decode.

Behavior summary:

- smaller value: ordinary speculative decode is triggered more easily;
- larger value: keep trying parallel speculative decode longer.

Typical use:

- use a smaller value when accept rate is small or often rollback.

#### `--remote-speculative-zmq-addr`

Meaning:

- The address used by SPECTRE ZMQ communication.

#### `--remote-speculative-zmq-port`

Meaning:

- The ZMQ port used between target and draft.

Requirement:

- target and draft must use the same value.

### 6.2 General speculative parameters used by SPECTRE

#### `--speculative-algorithm REMOTE`

Meaning:

- Enables the SPECTRE speculative path.

Without this, the server will not enter the `REMOTE` worker/scheduler logic.

#### `--speculative-num-steps`

Meaning:

- Controls the draft step depth.

In SPECTRE, this contributes to the draft tree depth and affects how many speculative verification steps the target prepares for.

Typical use:

- larger value: potentially more speculative gain, but more draft work and verification complexity;
- smaller value: simpler and usually more stable.

#### `--speculative-eagle-topk`

Meaning:

- Controls branch width per step.

Although SPECTRE is not local EAGLE, it still reuses the tree-based speculative verification structure, so this parameter still matters for tree shape.

Typical use:

- `1`: single-path draft, simplest and lowest communication overhead;
- `>1`: tree-shaped speculative candidates, higher complexity and potentially higher gain.

#### `--speculative-num-draft-tokens`

Meaning:

- Controls how many draft tokens are verified in one speculative round.

In current logic, you will often want it to match the expected tree size for your chosen `num_steps` and `topk`. A common simple setup is:

```text
speculative-num-steps = 3
speculative-eagle-topk = 1
speculative-num-draft-tokens = 4
```

## 7. Smoke Test

After both services start, send a simple health check to the target server:

```bash
curl http://127.0.0.1:30000/health
```

Then try a generate request:

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

If this works, the target side is serving correctly.  
If you do not see speculative benefit, then check:

- target logs;
- draft logs;
- ZMQ address and port;
- whether both sides were started with `--speculative-algorithm REMOTE`;
- whether one side is entering high-overhead fallback too often.

## 8. Benchmark

The repo already contains some benchmark helpers under `SPECTRE-pressure/`, for example:

- `SPECTRE-pressure/start_pressure.sh`

Typical flow:

1. Start the draft server.
2. Start the target server.
3. Run benchmark scripts against the target server HTTP endpoint.

## 9. Result Figure

Please replace the placeholder below after the final figure is added to the repo.

```markdown
![SPECTRE Result](./images/remote_spec_result_placeholder.png)
```

Or, if you prefer a visible TODO block:

```text
[TODO] Insert final SPECTRE result figure here.
```
