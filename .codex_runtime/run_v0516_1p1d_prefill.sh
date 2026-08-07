#!/usr/bin/env bash
set -euo pipefail

export MC_INTRANODE_NVLINK=1
export MOONCAKE_PROTOCOL=nvlink_intra
export SGLANG_MOONCAKE_SEND_AUX_TCP=0
export SGLANG_MOONCAKE_CUSTOM_MEM_POOL=INTRA_NODE_NVLINK
export SGLANG_DSPARK_PD_TARGET_LAYER_IDS=40,41,42
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1
export SGLANG_DG_CACHE_DIR=/ufs/zhangyu/deep_gemm/DeepSeek-V4-Flash-prefill/
export SGLANG_DSV4_FP4_EXPERTS=0
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024
export SGLANG_ENABLE_UNIFIED_RADIX_TREE=1
export SGLANG_OPT_FUSE_MHC_POST_PRE=1
export SGLANG_OPT_USE_CP_MULTI_STREAM_OVERLAP=0
export SGLANG_USE_TRITON_MOE_FUSED_GATE=1
export SGLANG_FLASHINFER_AUTOTUNE_CACHE=1
export SGLANG_DSV4_COMPRESS_STATE_DTYPE=bf16
export SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=12
export SGLANG_DISAGGREGATION_QUEUE_SIZE=3
export SGLANG_OPT_BF16_FP32_GEMM_ALGO=deep_gemm
export SGLANG_OPT_USE_ONLINE_COMPRESS=0
export SGLANG_EXPERIMENTAL_ONLINE_C128_MTP=0
export SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE=1
export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=32768
export JD_ENABLE_IGNORE_EOS=true

MODEL_PATH=/ufs/models/deepseek-ai/DeepSeek-V4-Flash-0731-FP8/
MODEL_CHECKSUM=/ufs/zhangyu/tmp/codex-dspark-hidden-opti-v0516-20260804/artifacts/model-checksums/fp8.json
ARTIFACT_DIR="${ARTIFACT_DIR:?ARTIFACT_DIR must be set}"
mkdir -p "${ARTIFACT_DIR}"

CUDA_VISIBLE_DEVICES=4,5,6,7 \
SGLANG_TORCH_PROFILER_DIR="${ARTIFACT_DIR}/profile-p0" \
sglang serve \
  --model-path "${MODEL_PATH}" \
  --model-checksum "${MODEL_CHECKSUM}" \
  --served-model-name DeepSeek-V4-Flash-3P1D \
  --trust-remote-code \
  --host 6.200.20.13 \
  --port 8182 \
  --dist-init-addr 6.200.20.13:8326 \
  --watchdog-timeout 3600 \
  --model-loader-extra-config '{"enable_multithread_load":"true","num_threads":64}' \
  --enable-metrics \
  --enable-cache-report \
  --enable-mfu-metrics \
  --tp-size 4 \
  --dp-size 1 \
  --enable-dsa-prefill-context-parallel \
  --dsa-prefill-cp-mode round-robin-split \
  --moe-dense-tp-size 1 \
  --mem-fraction-static 0.8 \
  --swa-full-tokens-ratio 0.02 \
  --context-length 1048576 \
  --kv-cache-dtype fp8_e4m3 \
  --max-running-requests 32 \
  --cuda-graph-max-bs-decode 32 \
  --chunked-prefill-size 32768 \
  --max-prefill-tokens 32768 \
  --load-balance-method follow_bootstrap_room \
  --tool-call-parser deepseekv4 \
  --reasoning-parser deepseek-v4 \
  --attention-backend dsv4 \
  --moe-runner-backend deep_gemm \
  --speculative-algorithm DSPARK \
  --speculative-num-steps 1 \
  --speculative-eagle-topk 1 \
  --speculative-dspark-block-size 5 \
  --enable-hierarchical-cache \
  --hicache-ratio 6 \
  --hicache-size 0 \
  --hicache-write-policy write_through \
  --hicache-io-backend direct \
  --hicache-mem-layout page_first_direct \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-ib-device mlx5_gdr_0,mlx5_gdr_1,mlx5_gdr_2,mlx5_gdr_3,mlx5_gdr_4,mlx5_gdr_5,mlx5_gdr_6,mlx5_gdr_7 \
  --disaggregation-bootstrap-port 18956 \
  2>&1 | tee "${ARTIFACT_DIR}/prefill-p0.log"
