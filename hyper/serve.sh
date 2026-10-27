#!/usr/bin/env bash
# serve.sh: start the API server (127.0.0.1:18020) for one installed model.
#
#   bash serve.sh --list             # models (models.sh); install with install.sh <id>
#   bash serve.sh                    # default model for this GPU (or $MODEL_ID)
#   bash serve.sh qwen3.6-35b-a3b    # llama.cpp llama-server  (serve-gguf.sh)
#   bash serve.sh qwen3.8-27b        # HyperQwen / vLLM        (this file)
#
# Both serve the OpenAI API (/v1/chat/completions) and the Anthropic API (/v1/messages)
# under the model id as the model name. One model at a time: they share the port and GPU.
#
# HyperQwen profiles, picked from the GPU (PROFILE=full|small to force):
# full  (>= 22 GB VRAM, Ampere+): HyperQwen's stock path, SPEC=mtp CTX=fast, 64k context.
# small (< 22 GB or pre-Ampere):  fp16, no speculation, Triton attention, part of the
#        weights in pinned host RAM (--cpu-offload-gb). Works, but every token streams
#        the offloaded weights over PCIe, so expect ~1 tok/s on an 8 GB laptop GPU.
#
# Any HyperQwen knob still applies (MAX_LEN, PORT, VLLM_API_KEY, EXTRA_ARGS, ...).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=models.sh
source "$HERE/models.sh"
case "${1:-}" in
  --list) echo "Models:"; model_list; exit 0 ;;
  -h|--help) sed -n '2,19p' "$0"; exit 0 ;;
esac
model_pick "${1:-}" || exit 2
[ "$M_BACKEND" = gguf ] && exec bash "$HERE/serve-gguf.sh" "$M_ID"
HQ_DIR="${HQ_DIR:-$HERE/HyperQwen}"
MODEL_REPO="${MODEL_REPO:-$M_REPO}"
export MODEL="${MODEL:-$HQ_DIR/models/${MODEL_REPO##*/}}"
[ -f "$MODEL/config.json" ] || { echo "model not found at $MODEL: run: bash install.sh $M_ID" >&2; exit 1; }

# Prepared weights with VISION=0 (tower dropped), measured from the files: ~15 GB.
WEIGHTS_GB="${WEIGHTS_GB:-15}"
# VRAM kept for CUDA context, the Marlin repack scratch at load, activations and KV.
# 2 failed on an 8 GB card (repack OOM); 4 boots with a 2.9 GiB / 28k-token KV pool.
RESERVE_GB="${RESERVE_GB:-4}"

IFS=',' read -r CAP MEM_MIB < <(nvidia-smi --query-gpu=compute_cap,memory.total \
  --format=csv,noheader,nounits | head -n1)
CAP="${CAP// /}"; MEM_GB=$(( ${MEM_MIB// /} / 1024 ))

if [ -z "${PROFILE:-}" ]; then
  if [ "$MEM_GB" -ge 22 ] && awk -v c="$CAP" 'BEGIN{exit !(c >= 8.0)}'; then PROFILE=full; else PROFILE=small; fi
fi

export VISION="${VISION:-0}"
case "$PROFILE" in
  full)
    export SPEC="${SPEC:-mtp}" CTX="${CTX:-fast}"
    ;;
  small)
    OFFLOAD_GB="${OFFLOAD_GB:-$(( WEIGHTS_GB + RESERVE_GB - MEM_GB ))}"
    [ "$OFFLOAD_GB" -lt 0 ] && OFFLOAD_GB=0
    export SPEC=off CTX=fast
    # The VMM allocator made gptq_marlin_repack fail ("aten::empty ... API call failed").
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}"
    # 28000: the most the ~28.4k-token KV pool on an 8 GB card holds, and enough for
    # Claude Code's ~15-25k-token system prompt (claude-qwen.sh).
    export MAX_LEN="${MAX_LEN:-28000}" MAX_SEQS="${MAX_SEQS:-1}" GPU_UTIL="${GPU_UTIL:-0.92}"
    # Later flags win over the launcher's CTX=fast defaults (FLASH_ATTN + bf16 KV),
    # neither of which exists on Turing.
    SMALL_ARGS="--dtype float16 --attention-backend TRITON_ATTN --kv-cache-dtype auto --enforce-eager"
    [ "$OFFLOAD_GB" -gt 0 ] && SMALL_ARGS="$SMALL_ARGS --cpu-offload-gb $OFFLOAD_GB"
    export EXTRA_ARGS="$SMALL_ARGS ${EXTRA_ARGS:-}"
    ;;
  *) echo "PROFILE must be full or small" >&2; exit 2 ;;
esac

echo "serve.sh: GPU sm${CAP/./} ${MEM_GB} GB -> PROFILE=$PROFILE SPEC=$SPEC CTX=$CTX ${OFFLOAD_GB:+OFFLOAD_GB=$OFFLOAD_GB}"
echo "serve.sh: MODEL=$MODEL"
cd "$HQ_DIR"
exec bash single-user/start_qwen.sh
