#!/usr/bin/env bash
# install-gguf.sh: install llama.cpp (prebuilt release) and download a GGUF model from models.sh.
# Usually called by install.sh. Idempotent: re-running skips done steps and resumes downloads.
#
#   bash install-gguf.sh qwen3.6-35b-a3b [--no-model]
#
# Env overrides:
#   LLAMA_TAG   llama.cpp release tag (default: the tested one)
#   LLAMA_DIR   where llama.cpp goes   (default: ./llama.cpp)
#   GGUF_DIR    where models go        (default: ./models)
#   GGUF_FILE   a different file from the same HF repo (e.g. the Q3_K quant)
#   LLAMA_FLAVOR cuda-12.8 | cuda-13.4 | vulkan | cpu  (default: cuda-12.8 if nvidia-smi works)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=models.sh
source "$HERE/models.sh"
WITH_MODEL=1
ARG_MODEL=""
for a in "$@"; do
  case "$a" in
    --no-model) WITH_MODEL=0 ;;
    -*) echo "unknown option $a" >&2; exit 2 ;;
    *) ARG_MODEL="$a" ;;
  esac
done
model_pick "$ARG_MODEL" || exit 2
[ "$M_BACKEND" = gguf ] || { echo "$M_ID is not a gguf model (backend $M_BACKEND); use install.sh" >&2; exit 2; }

# Release these scripts were tested with (2026-10-27). Newer ones usually work too.
LLAMA_TAG="${LLAMA_TAG:-b11934}"
LLAMA_DIR="${LLAMA_DIR:-$HERE/llama.cpp}"
GGUF_DIR="${GGUF_DIR:-$HERE/models}"
GGUF_FILE="${GGUF_FILE:-$M_FILE}"
MODEL_PATH="$GGUF_DIR/${M_REPO##*/}/$GGUF_FILE"
GH="https://github.com/ggml-org/llama.cpp/releases/download/$LLAMA_TAG"

log()  { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33mWARN: %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

preflight() {
  log "Preflight"
  for c in curl tar python3 sha256sum; do command -v "$c" >/dev/null || die "$c is required"; done
  [ "$(uname -m)" = x86_64 ] || die "only x86_64 Linux is scripted (got $(uname -m))"
  if [ -z "${LLAMA_FLAVOR:-}" ]; then
    if nvidia-smi >/dev/null 2>&1; then LLAMA_FLAVOR=cuda-12.8; else LLAMA_FLAVOR=cpu; fi
  fi
  nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/GPU: /' || echo "GPU: none (CPU only, slow)"
  echo "RAM: $(free -g | awk '/Mem:/{print $2}') GB   llama.cpp: $LLAMA_TAG ($LLAMA_FLAVOR)"
  echo "Disk free: $(df -BG --output=avail "$HERE" | tail -1 | tr -d ' ')"
}

# Fetch a release tarball and unpack it into $1 (flattens the top-level directory).
fetch_release() {
  local name="$1" dst="$2" tmp
  tmp="$LLAMA_DIR/dl"; mkdir -p "$tmp" "$dst"
  curl -fL -C - --retry 10 --retry-delay 5 -o "$tmp/$name" "$GH/$name"
  tar xzf "$tmp/$name" -C "$dst" --strip-components=1
}

install_llamacpp() {
  local dst="$LLAMA_DIR/$LLAMA_TAG-$LLAMA_FLAVOR" asset
  if [ -x "$dst/llama-server" ] && LD_LIBRARY_PATH="$dst" "$dst/llama-server" --version >/dev/null 2>&1; then
    log "llama.cpp $LLAMA_TAG ($LLAMA_FLAVOR) already installed"
  else
    case "$LLAMA_FLAVOR" in
      cuda-*) asset="llama-$LLAMA_TAG-bin-ubuntu-$LLAMA_FLAVOR-x64.tar.gz" ;;
      vulkan) asset="llama-$LLAMA_TAG-bin-ubuntu-vulkan-x64.tar.gz" ;;
      cpu)    asset="llama-$LLAMA_TAG-bin-ubuntu-x64.tar.gz" ;;
      *) die "LLAMA_FLAVOR must be cuda-12.8 | cuda-13.4 | vulkan | cpu" ;;
    esac
    log "Downloading llama.cpp $asset"
    fetch_release "$asset" "$dst"
    # The CUDA backend (libggml-cuda.so) links the CUDA runtime/cuBLAS; if the host lacks
    # them, add the matching cudart bundle next to the binaries.
    if [[ "$LLAMA_FLAVOR" == cuda-* ]] && LD_LIBRARY_PATH="$dst" ldd "$dst/libggml-cuda.so" | grep -q 'not found'; then
      log "CUDA runtime libs missing on this host: adding the cudart bundle"
      fetch_release "cudart-llama-$LLAMA_TAG-bin-ubuntu-$LLAMA_FLAVOR-x64.tar.gz" "$dst"
    fi
    LD_LIBRARY_PATH="$dst" "$dst/llama-server" --version || die "llama-server does not run (see above)"
    rm -rf "$LLAMA_DIR/dl"
  fi
  if [[ "$LLAMA_FLAVOR" == cuda-* ]]; then
    LD_LIBRARY_PATH="$dst" "$dst/llama-server" --list-devices 2>&1 | grep -q CUDA0 \
      || warn "llama-server sees no CUDA device: it will run on CPU only"
  fi
  ln -sfn "$(basename "$dst")" "$LLAMA_DIR/current"
}

download_model() {
  [ "$WITH_MODEL" = 1 ] || { log "Skipping model (--no-model)"; return; }
  mkdir -p "$(dirname "$MODEL_PATH")"
  if [ -f "$MODEL_PATH" ]; then
    log "Model already at ${MODEL_PATH#$HERE/}"
    return
  fi
  local url="https://huggingface.co/$M_REPO/resolve/main/$GGUF_FILE" want
  log "Downloading $M_REPO/$GGUF_FILE (resumable)"
  curl -fL -C - --retry 20 --retry-delay 5 -o "$MODEL_PATH.part" "$url"
  want=$(curl -fsS "https://huggingface.co/api/models/$M_REPO/tree/main" | python3 -c '
import json, sys
for f in json.load(sys.stdin):
    if f["path"] == sys.argv[1]: print((f.get("lfs") or {}).get("oid", ""))' "$GGUF_FILE")
  if [ -n "$want" ]; then
    log "Checking sha256 (a few minutes for ~20 GB)"
    [ "$(sha256sum "$MODEL_PATH.part" | cut -d' ' -f1)" = "$want" ] \
      || die "sha256 mismatch: delete $MODEL_PATH.part and rerun"
  else
    warn "no sha256 published for $GGUF_FILE; skipping the check"
  fi
  mv "$MODEL_PATH.part" "$MODEL_PATH"
}

preflight
install_llamacpp
download_model
log "Done. Start the server with:  bash $HERE/serve.sh $M_ID   then test with:  bash $HERE/test.sh"
