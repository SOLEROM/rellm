# shellcheck shell=bash
# models.sh: the models install.sh / serve.sh know about. Sourced, not run.
# shellcheck disable=SC2034  # M_* are read by the scripts that source this file
#
# One line per model:  id | backend | HF repo | file (gguf only) | description
#   backend hyperqwen: HyperQwen (patched vLLM), W4A16 safetensors, best on 24 GB Ampere+.
#   backend gguf:      llama.cpp llama-server, GGUF file, runs well on small GPUs + RAM.
# To add a model, add a line; the id is also the name the server answers to.
MODEL_TABLE='
qwen3.8-27b|hyperqwen|nchapman/Huihui-Qwen3.8-27B-abliterated-W4A16-GPTQ||Huihui Qwen3.8 27B abliterated, W4A16 (19.5 GB). ~100 tok/s on 24 GB Ampere+, ~1 tok/s on 8 GB
qwen3.6-35b-a3b|gguf|huihui-ai/Huihui-Qwen3.6-35B-A3B-abliterated-MTP-GGUF|Huihui-Qwen3.6-35B-A3B-abliterated-ggml-model-Q4_K.gguf|Huihui Qwen3.6 35B-A3B MoE abliterated, GGUF Q4_K (20 GB). 3B active: fast on 8 GB GPU + 32 GB RAM
'

model_list() {
  local def; def=$(model_default)
  echo "$MODEL_TABLE" | while IFS="|" read -r id backend _repo _file desc; do
    [ -n "$id" ] || continue
    printf '  %-18s %-10s %s%s\n' "$id" "$backend" "$desc" "$([ "$id" = "$def" ] && echo '  [default here]')"
  done
}

# model_get <id>: sets M_ID M_BACKEND M_REPO M_FILE M_DESC; returns 1 if unknown.
model_get() {
  local line
  line=$(echo "$MODEL_TABLE" | awk -F'|' -v id="$1" '$1==id' | head -n1)
  [ -n "$line" ] || return 1
  IFS='|' read -r M_ID M_BACKEND M_REPO M_FILE M_DESC <<<"$line"
}

# Default for this host: the 27B on a 24 GB Ampere+ card, the MoE GGUF anywhere else.
model_default() {
  local cap mem
  IFS=',' read -r cap mem < <(nvidia-smi --query-gpu=compute_cap,memory.total \
    --format=csv,noheader,nounits 2>/dev/null | head -n1) || true
  cap="${cap// /}"; mem="${mem// /}"
  if [ -n "$mem" ] && [ "$mem" -ge 22000 ] && awk -v c="$cap" 'BEGIN{exit !(c >= 8.0)}'; then
    echo qwen3.8-27b
  else
    echo qwen3.6-35b-a3b
  fi
}

# model_pick <arg-or-empty>: resolve from the arg, then $MODEL_ID, then the host default.
model_pick() {
  local want="${1:-${MODEL_ID:-$(model_default)}}"
  if ! model_get "$want"; then
    echo "unknown model '$want'. Known models:" >&2; model_list >&2; return 1
  fi
}
