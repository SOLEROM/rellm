#!/usr/bin/env bash
# serve.sh — start HyperQwen's single-user server on the Huihui 27B model, with a
# profile picked from the GPU it finds.
#
#   bash serve.sh               # foreground; Ctrl-C stops it
#   PROFILE=small bash serve.sh # force a profile: full | small
#
# full  (>= 22 GB VRAM, Ampere+): HyperQwen's stock path, SPEC=mtp CTX=fast, 64k context.
# small (< 22 GB or pre-Ampere):  fp16, no speculation, Triton attention, part of the
#        weights in pinned host RAM (--cpu-offload-gb). Works, but every token streams
#        the offloaded weights over PCIe, so expect ~1 tok/s on an 8 GB laptop GPU.
#
# Any HyperQwen knob still applies (MAX_LEN, PORT, VLLM_API_KEY, EXTRA_ARGS, ...).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HQ_DIR="${HQ_DIR:-$HERE/HyperQwen}"
MODEL_REPO="${MODEL_REPO:-nchapman/Huihui-Qwen3.8-27B-abliterated-W4A16-GPTQ}"
export MODEL="${MODEL:-$HQ_DIR/models/${MODEL_REPO##*/}}"
[ -f "$MODEL/config.json" ] || { echo "model not found at $MODEL: run install.sh first" >&2; exit 1; }

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
