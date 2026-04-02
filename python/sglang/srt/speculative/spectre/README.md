# SPECTRE

**SPECTRE** (**Spec**ulative decoding with a **Re**mote drafter) is SGLang's remote speculative decoding mode.

It separates the speculative pipeline into two independently deployed services:

- **Verifier** (`--spectre-role target`): runs the large model, serves user traffic, verifies draft tokens.
- **Drafter** (`--spectre-role draft`): runs the small model, generates draft tokens on behalf of the Verifier.

Compared with local speculative decoding, SPECTRE offloads draft generation to a separate process or machine. This is useful when:

- the large model is GPU-memory-bound and cannot co-host a draft model;
- the small model can run on a separate GPU or node;
- you want to decouple draft throughput from verification throughput independently.

---

## Table of Contents

1. [Architecture](#1-architecture)
2. [Code Structure](#2-code-structure)
3. [Environment Setup](#3-environment-setup)
4. [Core Concepts](#4-core-concepts)
5. [How To Start SPECTRE](#5-how-to-start-spectre)
6. [Args](#6-args)
7. [Smoke Test](#7-smoke-test)
8. [Benchmark](#8-benchmark)

---

## 1. Architecture

### 1.1 Data flow

```
  Client
    │  HTTP / OpenAI-compatible
    ▼
┌─────────────────────────┐
│   Verifier (large model) │
│  SpectreTargetScheduler  │
│  ┌───────────────────┐   │
│  │  DraftCircuitBreaker│  │
│  └───────────────────┘   │
└──────────┬───────────────┘
           │ ZMQ (Router ↔ Dealer)
           │ TCP (cross-machine) or IPC (same-machine, auto)
           ▼
┌─────────────────────────┐
│   Drafter (small model)  │
│  SpectreDraftScheduler   │
│  ┌───────────────────┐   │
│  │SpectreDraftStateManager│
│  └───────────────────┘   │
└─────────────────────────┘
```

### 1.2 Request lifecycle

| Step | Who | What |
|---|---|---|
| 1 | Client | Sends generation request to Verifier over HTTP |
| 2 | Verifier | On each decode step, sends a `SpectreRequest(action=DRAFT)` to Drafter via ZMQ |
| 3 | Drafter | Runs small model, returns `SpectreRequest(action=DRAFT, draft_token_ids=[...])` |
| 4 | Verifier | Verifies draft tokens with the large model; accepts valid prefix, rejects the rest |
| 5 | Verifier | When request finishes or aborts, sends `SpectreRequest(action=FINISH/ABORT)` to Drafter |
| 6 | Client | Receives final generation result from Verifier |

### 1.3 ZMQ transport selection

`SpectreConfig` auto-selects the transport based on `--spectre-zmq-addr`:

| `zmq-addr` value | Transport | Socket path |
|---|---|---|
| `127.0.0.1` or `0.0.0.0` | **IPC** (faster, no network stack) | `/tmp/<addr>_<port>` |
| Any other IP | **TCP** | `tcp://<addr>:<port>` |

For same-machine deployments, IPC is automatically used even if you pass `--spectre-zmq-addr 127.0.0.1`.

### 1.4 ZMQ socket roles

- **Verifier** uses a **Router** socket (binds, accepts connections from multiple Drafters).
- **Drafter** uses a **Dealer** socket (connects to the Router, auto-generates a unique identity).

This means one Verifier can be connected to multiple Drafter instances (though the current routing logic uses the first available Drafter identity).

### 1.5 Protocol messages

All messages are `SpectreRequest` dataclass objects serialized via msgpack through the C++ ZMQ layer.

Key fields:

| Field | Type | Description |
|---|---|---|
| `request_id` | `str` | SGLang request ID (`req.rid`) |
| `spec_cnt` | `int` | Speculative round counter, monotonically increasing per request. Used to correlate requests/responses and filter stale messages. |
| `action` | `SpectreAction` | `DRAFT` / `FINISH` / `ABORT` / `REJECT` |
| `spec_type` | `SpecType` | `DRAFT_REQUEST` (T→D) / `DRAFT_RESPONSE` (D→T) |
| `draft_token_ids` | `List[int]` | Draft tokens proposed by the Drafter |
| `input_ids` | `List[int]` | Full input token IDs (sent on `spec_cnt == 0` only) |
| `output_ids` | `List[int]` | Tokens generated so far (sent every round) |
| `num_draft_tokens` | `int` | How many draft tokens the Verifier is requesting |
| `sampling_params` | `SamplingParams` | Sampling configuration (sent on `spec_cnt == 0` only) |

Action semantics:

| `SpectreAction` | Direction | Meaning |
|---|---|---|
| `DRAFT` | T→D and D→T | Normal draft request / draft response |
| `FINISH` | T→D | Request completed normally; Drafter should clean up its state |
| `ABORT` | T→D | Request aborted; Drafter should clean up its state |
| `REJECT` | D→T | Drafter is overloaded; Verifier should temporarily stop sending |

---

## 2. Code Structure

```
python/sglang/srt/speculative/spectre/
│
├── spectre_protocol.py          # SpectreRequest, SpectreAction, SpecType
├── spectre_communication.py     # SpectreConfig, SpectreZMQCommunicator
│
├── verifier/                    # Target (large model) side
│   ├── spectre_target_scheduler_mixin.py   # SpectreTargetSchedulerMixin, DraftCircuitBreaker
│   └── spectre_worker.py                  # Worker thread / TP coordination
│
├── drafter/                     # Draft (small model) side
│   ├── spectre_draft_scheduler_mixin_v2.py # SpectreDraftSchedulerMixinV2 (main scheduler loop)
│   ├── spectre_draft_scheduler_mixin.py    # Legacy v1 mixin (kept for reference)
│   ├── spectre_state_manager.py            # SpectreDraftState, SpectreDraftStateManager
│   └── spectre_kv_rollbacker.py            # SpectreKVRollbacker (KV cache rollback on rejection)
│
└── cpp_zmq/                     # C++ ZMQ extension (pybind11)
    ├── include/
    │   ├── spectre_protocol.hpp
    │   ├── spectre_zmq_endpoints.hpp
    │   ├── spectre_zmq_logging.hpp
    │   └── spectre_zmq_serialization.hpp
    ├── src/
    │   ├── spectre_zmq.cpp              # RouterEndpoint, DealerEndpoint
    │   ├── spectre_zmq_endpoints.cpp
    │   ├── spectre_zmq_logging.cpp      # Reads SPECTRE_DEBUG env var
    │   └── spectre_zmq_serialization.cpp
    ├── scripts/build_cpp_zmq.sh
    └── setup.py
```

Integration points in the broader SGLang codebase:

| File | Role |
|---|---|
| `srt/server_args.py` | Defines all `spectre_*` CLI args and dataclass fields |
| `srt/speculative/spec_info.py` | Registers `SPECTRE` algorithm type, routes to SPECTRE event loop |
| `srt/managers/scheduler.py` | Mixes in `SpectreTargetSchedulerMixin` or `SpectreDraftSchedulerMixinV2` based on role |

---

## 3. Environment Setup

### 3.1 Prerequisites

This SPECTRE implementation is developed against `SGLang 0.5.10rc`.

Recommended environment:

- Linux
- Python `>= 3.10`
- CUDA-enabled PyTorch
- At least 2 GPUs (or 2 hosts) that can communicate over the configured ZMQ port

For cross-machine deployment, ensure:

- The Verifier host can reach the Drafter host on the configured ZMQ port.
- Firewalls and security groups allow that traffic.
- Both hosts can access their respective model paths.

### 3.2 Build the C++ ZMQ extension

SPECTRE's ZMQ communication relies on a C++ pybind11 extension. Build it once before starting any server:

```bash
cd python/sglang/srt/speculative/spectre/cpp_zmq
pip install pybind11 msgpack
apt-get update && apt-get install -y libmsgpack-dev
python3 setup.py build_ext --inplace
```

Or use the provided script:

```bash
bash python/sglang/srt/speculative/spectre/cpp_zmq/scripts/build_cpp_zmq.sh
```

### 3.3 Verify basic serving first

Before enabling SPECTRE, confirm a plain server starts correctly:

```bash
python -m sglang.launch_server \
  --model-path /path/to/model \
  --host 0.0.0.0 \
  --port 30000
```

If this fails, fix the base environment first.

---

## 4. Core Concepts

### 4.1 Verifier (target server)

The Verifier:

- loads the large model;
- receives user requests over HTTP;
- on each decode step, sends draft requests to the Drafter via ZMQ;
- waits up to `SPECTRE_RECV_TIMEOUT_MS` ms for draft tokens to arrive;
- verifies draft tokens with the large model and accepts the valid prefix;
- returns final generation results to the client.

Only the Verifier is exposed to clients. The Drafter is an internal backend service.

Key class: `SpectreTargetSchedulerMixin` in `verifier/spectre_target_scheduler_mixin.py`.

### 4.2 Drafter (draft server)

The Drafter:

- loads the small model;
- receives draft requests from the Verifier via ZMQ;
- maintains per-request draft state (`SpectreDraftStateManager`) to track where each request is in the generation pipeline;
- runs the small model and sends draft tokens back;
- sends `action=REJECT` when its running batch is too large (`spectre_max_batch_size`), signalling the Verifier to back off.

Key class: `SpectreDraftSchedulerMixinV2` in `drafter/spectre_draft_scheduler_mixin_v2.py`.

### 4.3 Circuit breaker

The Verifier maintains a `DraftCircuitBreaker` to handle situations where the Drafter is unresponsive or offline. It operates in three states:

| State | Behavior |
|---|---|
| `CLOSED` | Normal operation. Draft requests are sent every round. |
| `OPEN` | Too many consecutive timeouts. Verifier stops sending draft requests and falls back to single-token decoding for `SPECTRE_COOLDOWN_ROUNDS` rounds. |
| `HALF_OPEN` | Cooldown period expired. Verifier sends one probe request. If successful, transitions back to `CLOSED`; if it fails again, returns to `OPEN`. |

Transitions are logged at `INFO` level with `[CircuitBreaker]` prefix.

### 4.4 Draft-Priority scheduling mode

Enabled with `--spectre-draft-priority` on the **Drafter**.

In the default (non-priority) mode, incoming draft requests from the Verifier are mixed into the Drafter's normal running batch. Under high load, normal requests can crowd out draft tokens, causing the Verifier to time out repeatedly.

In Draft-Priority mode:

1. The Drafter runs a **dedicated draft batch** for up to `--spectre-max-draft-priority-steps` steps before processing normal requests.
2. This guarantees that draft tokens are generated promptly before the Verifier's `SPECTRE_RECV_TIMEOUT_MS` deadline.
3. `--spectre-max-draft-priority-steps 0` means the step count is auto-computed from the maximum remaining draft steps across all active requests.

### 4.5 Speculative round counter (`spec_cnt`)

Each request has a `spec_cnt` field that increments by 1 on each speculative decode round. It is used to:

- correlate a Drafter response with the correct Verifier request;
- discard stale messages that arrive after a flush or restart;
- clean up stale KV cache state when speculative rounds are skipped.

### 4.6 KV cache rollback

On the Drafter side, `SpectreKVRollbacker` manages KV cache state when a speculative sequence is rejected. When the Verifier rejects a draft suffix, the Drafter must roll back its KV cache to the last accepted token before accepting new draft requests for that sequence.

### 4.7 Draft state management

`SpectreDraftStateManager` tracks the lifecycle of each request on the Drafter side:

- `SpectreDraftState` stores the request object, its current location (`draft_waiting`, `draft_batch`, `paused`), the last known prefix and output lengths, and a timestamp.
- A periodic cleanup sweep (every `SGLANG_DRAFT_CLEANUP_INTERVAL` forward cycles) removes states for requests that have not been updated for more than 60 seconds.

---

## 5. How To Start SPECTRE

Replace model paths, TP size, ports, and addresses with your own values.

### 5.1 Start the Drafter (small model)

```bash
python -m sglang.launch_server \
  --model-path /path/to/small-model \
  --tp 1 \
  --served-model-name small-model \
  --mem-fraction-static 0.8 \
  --host 0.0.0.0 \
  --port 3008 \
  --max-running-requests 256 \
  --chunked-prefill-size 16384 \
  --context-length 16384 \
  --trust-remote-code \
  --cuda-graph-max-bs 128 \
  --disable-overlap-schedule \
  --spectre-role draft \
  --spectre-zmq-port 30090 \
  --spectre-draft-priority \
  --spectre-max-draft-priority-steps 10 \
  --spectre-max-batch-size 3200 \
  --log-level info
```

### 5.2 Start the Verifier (large model)

```bash
python -m sglang.launch_server \
  --model-path /path/to/large-model \
  --tp 1 \
  --served-model-name large-model \
  --mem-fraction-static 0.8 \
  --host 0.0.0.0 \
  --port 30000 \
  --max-running-requests 256 \
  --chunked-prefill-size 16384 \
  --context-length 16384 \
  --trust-remote-code \
  --cuda-graph-max-bs 128 \
  --disable-overlap-schedule \
  --speculative-algorithm SPECTRE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --spectre-role target \
  --spectre-zmq-addr 127.0.0.1 \
  --spectre-zmq-port 30090 \
  --spectre-max-batch-size 1600 \
  --spectre-reject-interval 2000 \
  --spectre-retry-fail-ratio 0.6 \
  --spectre-retry-min-count 15 \
  --attention-backend fa3 \
  --log-level info
```

After both servers are ready, send all client traffic to the Verifier only:

```
http://127.0.0.1:30000
```

### 5.3 Cross-machine deployment

When Drafter and Verifier run on different machines:

1. Pass the Drafter machine's actual IP to the Verifier: `--spectre-zmq-addr <DRAFTER_IP>`.
2. ZMQ will automatically use TCP (not IPC).
3. Make sure `--spectre-zmq-port` matches on both sides and is open in the firewall.

---

## 6. Args

### 6.1 CLI Arguments

#### Shared (both Verifier and Drafter)

| CLI Argument | Type | Default | Description |
|---|---|---|---|
| `--spectre-role` | `str` | `None` | Role of this server: `target` (Verifier) or `draft` (Drafter). |
| `--spectre-zmq-addr` | `str` | `None` | ZMQ address. Use `127.0.0.1` for same-machine (auto-uses IPC); use the Drafter's IP for cross-machine (uses TCP). |
| `--spectre-zmq-port` | `str` | `None` | ZMQ port. Must match on both sides. |
| `--spectre-max-batch-size` | `int` | `32` | Overload threshold. When the running batch size exceeds this, the Drafter sends `REJECT` and the Verifier stops sending draft requests. |

#### Verifier-side Only

| CLI Argument | Type | Default | Description |
|---|---|---|---|
| `--speculative-algorithm` | `str` | `None` | Must be `SPECTRE` to activate SPECTRE on the Verifier. |
| `--speculative-num-steps` | `int` | `None` | Speculative steps per round (draft tree depth). |
| `--speculative-eagle-topk` | `int` | `None` | Branch width per speculative step (draft tree width). |
| `--speculative-num-draft-tokens` | `int` | `None` | Total draft tokens verified per round. Typical: `steps=3, topk=1, draft_tokens=4`. |
| `--spectre-reject-interval` | `int` | `500` | After receiving a `REJECT`, the Verifier skips sending draft requests for this many decode rounds before retrying. `1` means retry every round. |
| `--spectre-no-draft-ratio` | `float` | `0.5` | If the fraction of requests with no usable draft tokens exceeds this value, the Verifier falls back to single-token decoding for the entire batch. |
| `--spectre-retry-fail-ratio` | `float` | `0.5` | Minimum fraction of failed pre-verify requests in a batch required to trigger an inline retry. |
| `--spectre-retry-min-count` | `int` | `4` | Minimum absolute count of failed requests to trigger retry (prevents false triggers on tiny batches). |

#### Drafter-side Only

| CLI Argument | Type | Default | Description |
|---|---|---|---|
| `--spectre-draft-priority` | `flag` | `False` | Enable Draft-Priority scheduling. The Drafter runs a dedicated draft batch for N steps before processing normal requests, preventing draft-token starvation under load. |
| `--spectre-max-draft-priority-steps` | `int` | `0` | Maximum steps to run in Draft-Priority mode per iteration. `0` = auto-compute from maximum remaining steps across active draft requests. Only effective with `--spectre-draft-priority`. |

---

### 6.2 Environment Variables

Set before launching the process. They are for low-level tuning that does not belong in CLI arguments.

| Environment Variable | Side | Default | Description |
|---|---|---|---|
| `SPECTRE_RECV_TIMEOUT_MS` | Verifier | `200` | Maximum time (ms) the Verifier waits for draft tokens from the Drafter per ZMQ receive call. Increase for high-latency cross-machine links. |
| `SPECTRE_FAILURE_THRESHOLD` | Verifier | `30` | Consecutive rounds with no useful draft response before the circuit breaker trips to `OPEN` state. |
| `SPECTRE_COOLDOWN_ROUNDS` | Verifier | `100` | Decode rounds the circuit breaker stays `OPEN` before transitioning to `HALF_OPEN` and probing the Drafter again. |
| `SGLANG_DRAFT_CLEANUP_INTERVAL` | Drafter | `500` | Number of Drafter forward cycles between stale-state cleanup sweeps. States inactive for > 60 s are removed. Lower values increase cleanup frequency at a small overhead. |
| `SPECTRE_DEBUG` | Both | `0` | Set to `1` for info-level ZMQ logging; `2` for debug-level (includes per-message timing). Applied in both Python (`spectre_communication.py`) and C++ (`spectre_zmq_logging.cpp`). |

---

### 6.3 Quick-start defaults

Minimal speculative configuration for the Verifier:

```text
--speculative-num-steps 3
--speculative-eagle-topk 1
--speculative-num-draft-tokens 4
```

This produces a single draft path of depth 3, with 4 tokens verified per round.

---

## 7. Smoke Test

After both services start, verify the Verifier is healthy:

```bash
curl http://127.0.0.1:30000/health
```

Then send a simple generation request:

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

If you do not see speculative speedup, check:

- Verifier and Drafter logs for `[CircuitBreaker]` transitions — frequent `CLOSED → OPEN` means the Drafter is too slow or unreachable.
- ZMQ address and port match on both sides.
- `--speculative-algorithm SPECTRE` is set on the **Verifier** only (not the Drafter).
- `--spectre-max-batch-size` is not set too low, causing constant `REJECT` messages.
- `SPECTRE_DEBUG=1` on both sides to see ZMQ-level message counts and timing.

---

## 8. Benchmark

Benchmark helpers are under `SPECTRE-pressure/`:

```
SPECTRE-pressure/start_pressure.sh
```

Typical flow:

1. Start the Drafter.
2. Start the Verifier.
3. Run benchmark scripts against the Verifier HTTP endpoint.
4. Compare throughput (tokens/s) against a standalone Verifier without SPECTRE.

---

## 9. Result

```text
[TODO] Insert final SPECTRE result figure here.
```
