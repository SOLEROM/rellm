#!/usr/bin/env bash
# serve-gguf.sh: run llama.cpp's llama-server on a GGUF model from models.sh.
# Usually called by serve.sh. Serves the OpenAI API (/v1/chat/completions) and the
# Anthropic API (/v1/messages) on 127.0.0.1:$PORT, model name = the model id.
#
#   bash serve-gguf.sh qwen3.6-35b-a3b
#
# Env knobs:
#   PORT        18020
#   CTX_SIZE    context tokens (default 65536; Claude Code needs >= ~24k)
#   SPEC        mtp | off      multi-token-prediction speculative decoding (default mtp)
#   MTP_N       max draft tokens per step (default 6, per the model card)
#   CPU_MOE     MoE layers whose experts stay in RAM (default: auto-fit to free VRAM)
#   HOST        bind address (default 127.0.0.1; 0.0.0.0 only with an API key)
#   VLLM_API_KEY / api_key.txt   API key (same as HyperQwen)
#   EXTRA_ARGS  more llama-server flags, appended last
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=models.sh
source "$HERE/models.sh"
model_pick "${1:-}" || exit 2
[ "$M_BACKEND" = gguf ] || { echo "$M_ID is not a gguf model; use serve.sh" >&2; exit 2; }

LLAMA_DIR="${LLAMA_DIR:-$HERE/llama.cpp}"
GGUF_DIR="${GGUF_DIR:-$HERE/models}"
MODEL_PATH="$GGUF_DIR/${M_REPO##*/}/${GGUF_FILE:-$M_FILE}"
BIN="$LLAMA_DIR/current/llama-server"
[ -x "$BIN" ] || { echo "llama-server not found: run: bash install.sh $M_ID" >&2; exit 1; }
[ -f "$MODEL_PATH" ] || { echo "model not found at $MODEL_PATH: run: bash install.sh $M_ID" >&2; exit 1; }
export LD_LIBRARY_PATH="$LLAMA_DIR/current${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

PORT="${PORT:-18020}"
CTX_SIZE="${CTX_SIZE:-65536}"
SPEC="${SPEC:-mtp}"
KEY="${VLLM_API_KEY:-$(cat "$HERE/api_key.txt" "$HERE/HyperQwen/api_key.txt" 2>/dev/null | head -n1 || true)}"
HOST="${HOST:-127.0.0.1}"
[ "$HOST" != 127.0.0.1 ] && [ -z "$KEY" ] && { echo "refusing HOST=$HOST without an API key" >&2; exit 1; }

ARGS=(-m "$MODEL_PATH" --alias "$M_ID" --host "$HOST" --port "$PORT"
      -c "$CTX_SIZE" -np 1 --jinja -fa on
      --cache-type-k q8_0 --cache-type-v q8_0)
# MoE: attention + shared weights on the GPU, as many expert layers as fit; the rest in RAM.
# --fit picks the GPU layers / CPU-MoE split itself, but only if -ngl is left unset.
if [ -n "${CPU_MOE:-}" ]; then ARGS+=(-ngl 999 --n-cpu-moe "$CPU_MOE" --fit off); else ARGS+=(--fit on); fi
# Patched chat template, if the repo has one for this model (see templates/).
[ -f "$HERE/templates/$M_ID.jinja" ] && ARGS+=(--chat-template-file "$HERE/templates/$M_ID.jinja")
[ "$SPEC" = mtp ] && ARGS+=(--spec-type draft-mtp --spec-draft-n-max "${MTP_N:-6}")
[ -n "$KEY" ] && ARGS+=(--api-key "$KEY")
# shellcheck disable=SC2206
[ -n "${EXTRA_ARGS:-}" ] && ARGS+=($EXTRA_ARGS)

echo "serve-gguf.sh: $M_ID  ctx=$CTX_SIZE spec=$SPEC  ${CPU_MOE:+cpu-moe=$CPU_MOE }-> http://$HOST:$PORT"
echo "serve-gguf.sh: MODEL=$MODEL_PATH"
exec "$BIN" "${ARGS[@]}"
