#!/usr/bin/env bash
# install.sh — install HyperQwen (bare-metal venv) and prepare the Huihui
# Qwen3.8-27B abliterated model for it. Idempotent: re-running skips done steps.
#
#   bash install.sh                 # everything: venv, vLLM, patches, model, prepare
#   bash install.sh --no-model      # stack only (no 19.5 GB download)
#   SKIP_VERIFY=1 bash install.sh   # skip verify.sh
#
# Env overrides:
#   HQ_DIR      HyperQwen checkout   (default: ./HyperQwen next to this script)
#   HQ_REF      HyperQwen commit/branch (default: the tested commit)
#   MODEL_REPO  HF repo to serve     (default: nchapman/Huihui-Qwen3.8-27B-abliterated-W4A16-GPTQ)
#   HQ_EXTRAS   "fa2 pthfa kvarn"    optional CUDA builds (sm80+ only; off by default)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HQ_DIR="${HQ_DIR:-$HERE/HyperQwen}"
HQ_GIT="https://github.com/syv-ai/HyperQwen"
# Commit these scripts were tested against (2026-10-27). HQ_REF=main for the latest.
HQ_REF="${HQ_REF:-ce9615c4f2309be727407918a6c979bd67968069}"
MODEL_REPO="${MODEL_REPO:-nchapman/Huihui-Qwen3.8-27B-abliterated-W4A16-GPTQ}"
MODEL_DIR="$HQ_DIR/models/${MODEL_REPO##*/}"
VLLM_VERSION="0.31.0"
MIN_DISK_GB=45
WITH_MODEL=1
[ "${1:-}" = "--no-model" ] && WITH_MODEL=0

log()  { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33mWARN: %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

preflight() {
  log "Preflight"
  command -v nvidia-smi >/dev/null || die "nvidia-smi not found: install the NVIDIA driver first"
  command -v git >/dev/null || die "git is required"
  command -v python3.12 >/dev/null || die "python3.12 is required (vLLM $VLLM_VERSION pins it)"
  python3.12 -c 'import venv, ensurepip' 2>/dev/null || die "install python3.12-venv"

  local name cap mem
  IFS=',' read -r name cap mem < <(nvidia-smi --query-gpu=name,compute_cap,memory.total \
    --format=csv,noheader,nounits | head -n1)
  cap="${cap// /}"; mem="${mem// /}"
  echo "GPU: $name  (sm${cap/./}, ${mem} MiB)"
  awk -v c="$cap" 'BEGIN{exit !(c < 7.5)}' && die "compute capability $cap < 7.5: vLLM $VLLM_VERSION cannot run here"
  awk -v c="$cap" 'BEGIN{exit !(c < 8.0)}' && warn "sm$cap (pre-Ampere): no bf16; serve.sh will use fp16 + SPEC=off (untested upstream)"
  [ "$mem" -lt 22000 ] && warn "${mem} MiB VRAM < 24 GB: the 27B will need CPU weight offload (slow). See PLAN.md"

  local free_gb
  free_gb=$(df -BG --output=avail "$HERE" | tail -1 | tr -dc 0-9)
  echo "Disk free: ${free_gb} GB"
  [ "$free_gb" -lt "$MIN_DISK_GB" ] && die "need ~${MIN_DISK_GB} GB free disk"
  echo "RAM: $(free -g | awk '/Mem:/{print $2}') GB"
}

checkout() {
  if [ -d "$HQ_DIR/.git" ]; then
    log "HyperQwen already at $HQ_DIR ($(git -C "$HQ_DIR" rev-parse --short HEAD))"
  else
    log "Cloning HyperQwen @ ${HQ_REF:0:12}"
    git clone "$HQ_GIT" "$HQ_DIR"
    git -C "$HQ_DIR" checkout -q "$HQ_REF"
  fi
}

install_venv() {
  cd "$HQ_DIR"
  if venv/bin/python -c "import vllm,sys; sys.exit(vllm.__version__!='$VLLM_VERSION')" 2>/dev/null; then
    log "venv already has vLLM $VLLM_VERSION"
    return
  fi
  log "Creating venv + installing vLLM $VLLM_VERSION (several GB, ~10-20 min)"
  [ -x venv/bin/python ] || python3.12 -m venv venv
  venv/bin/pip install -q --upgrade pip
  # Same pins as docs/install.md: nvidia-* hold nvcc at the CUDA 13.0 runtime version.
  venv/bin/pip install "vllm==$VLLM_VERSION" huggingface_hub hf_transfer ninja pandas \
    --extra-index-url https://flashinfer.ai/whl/ flashinfer-cubin==0.7.0.post1 \
    nvidia-cuda-nvcc==13.0.88 nvidia-cuda-crt==13.0.88 nvidia-cuda-cccl==13.0.85 nvidia-nvvm==13.0.88
}

apply_patches() {
  cd "$HQ_DIR"
  local sp
  sp=$(venv/bin/python -c 'import vllm, os; print(os.path.dirname(vllm.__file__))' 2>/dev/null | tail -n1)
  [ -d "$sp" ] || die "cannot locate the vllm package in the venv"
  # verify.sh --install is the authority on whether every patch is applied.
  if bash verify.sh --install >/dev/null 2>&1; then
    log "Patches already applied"
    return
  fi
  log "Applying the HyperQwen patch series to $sp"
  bash patches/apply.sh "$sp"
}

install_extras() {
  cd "$HQ_DIR"
  for x in ${HQ_EXTRAS:-}; do
    case "$x" in
      fa2|pthfa|kvarn) log "Building optional $x"; bash "$x/install.sh" ;;
      *) warn "unknown extra '$x' (fa2|pthfa|kvarn)";;
    esac
  done
}

fetch_and_prepare_model() {
  cd "$HQ_DIR"
  [ "$WITH_MODEL" = 1 ] || { log "Skipping model (--no-model)"; return; }
  log "Downloading $MODEL_REPO -> ${MODEL_DIR#$HQ_DIR/}"
  HF_HUB_ENABLE_HF_TRANSFER=1 venv/bin/python prepare/fetch_thirdparty.py "$MODEL_REPO" "$MODEL_DIR"

  # The checkpoint ships a bf16 lm_head/embeddings/MTP; requantize them (CPU, streams the shard).
  # A requantized head is stored packed (lm_head.weight_packed); the original is lm_head.weight.
  if grep -q '"lm_head.weight_packed"' "$MODEL_DIR/model.safetensors.index.json" 2>/dev/null; then
    log "Heads already requantized"
  else
    log "Requantizing lm_head / embeddings / MTP (CPU only, ~15 min)"
    venv/bin/python prepare/quant_heads_stream.py "$MODEL_DIR"
  fi

  if [ -f "$MODEL_DIR/mtp_draft_vocab_ids.pt" ]; then
    log "Draft vocab already built"
  else
    log "Building the 40k draft vocabulary"
    venv/bin/python prepare/build_draft_vocab.py "$MODEL_DIR" --ids prepare/draft_vocab_ids.json
  fi
}

verify() {
  cd "$HQ_DIR"
  [ "${SKIP_VERIFY:-0}" = 1 ] && return
  log "verify.sh --install"
  bash verify.sh --install
  if [ "$WITH_MODEL" = 1 ]; then
    log "verify.sh --no-server (model checks)"
    MODEL="$MODEL_DIR" bash verify.sh --no-server || warn "some model checks failed (see above)"
  fi
}

main() {
  preflight
  checkout
  install_venv
  apply_patches
  install_extras
  fetch_and_prepare_model
  verify
  log "Done. Start the server with:  bash $HERE/serve.sh   then test with:  bash $HERE/test.sh"
}
main "$@"
