#!/usr/bin/env bash
# claude-qwen.sh: run Claude Code against the local server (serve.sh: HyperQwen/vLLM or
# llama.cpp) instead of Anthropic's API. Both serve the Anthropic Messages API at
# /v1/messages, so no proxy. Model name and context size are read from the server.
#
#   bash claude-qwen.sh                 # interactive
#   bash claude-qwen.sh -p "explain x"  # any claude CLI args pass through
#
# Needs the server running (bash serve.sh <model>) with a context >= ~24k: Claude Code's
# system prompt + tool definitions alone are ~15-25k tokens.
# Only sets env vars for this one process: ~/.claude and other claude sessions are untouched.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PORT:-18020}"
BASE="http://127.0.0.1:$PORT"
KEY="${VLLM_API_KEY:-$(cat "$HERE/api_key.txt" "$HERE/HyperQwen/api_key.txt" 2>/dev/null | head -n1 || true)}"
KEY="${KEY:-local}"

if [ "$(curl -s -o /dev/null -w '%{http_code}' "$BASE/health")" != 200 ]; then
  echo "HyperQwen server not reachable at $BASE — start it with: bash $HERE/serve.sh" >&2
  exit 1
fi

# Served model name + context window: vLLM reports max_model_len in /v1/models,
# llama-server reports n_ctx in /props.
read -r DET_NAME DET_CTX < <(
  { curl -s -H "Authorization: Bearer $KEY" "$BASE/v1/models"; echo; \
    curl -s -H "Authorization: Bearer $KEY" "$BASE/props"; } | python3 -c '
import json, sys
name, ctx = "", 0
for line in sys.stdin.read().splitlines():
    try: d = json.loads(line)
    except ValueError: continue
    if d.get("data"):
        name = name or d["data"][0].get("id", "")
        ctx = ctx or d["data"][0].get("max_model_len") or 0
    ctx = ctx or (d.get("default_generation_settings") or {}).get("n_ctx") or 0
print(name or "-", ctx or 0)' 2>/dev/null || echo "- 0")
MODEL_NAME="${MODEL_NAME:-$DET_NAME}"
[ "$MODEL_NAME" != - ] || { echo "could not read the model name from $BASE/v1/models" >&2; exit 1; }
CTX="${CLAUDE_CODE_MAX_CONTEXT_TOKENS:-$DET_CTX}"
[ "$CTX" -gt 0 ] 2>/dev/null || CTX=28000
echo "claude-qwen.sh: $MODEL_NAME at $BASE (context $CTX)" >&2

export ANTHROPIC_BASE_URL="$BASE"
export ANTHROPIC_AUTH_TOKEN="$KEY"
unset ANTHROPIC_API_KEY
# Every model tier Claude Code might pick maps to the one local model.
export ANTHROPIC_MODEL="$MODEL_NAME"
export ANTHROPIC_DEFAULT_OPUS_MODEL="$MODEL_NAME"
export ANTHROPIC_DEFAULT_SONNET_MODEL="$MODEL_NAME"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="$MODEL_NAME"
export ANTHROPIC_SMALL_FAST_MODEL="$MODEL_NAME"
export CLAUDE_CODE_SUBAGENT_MODEL="$MODEL_NAME"
# Real window of the local server, so Claude Code compacts before the server rejects
# the prompt; also silences the "doesn't recognize the model" warning.
export CLAUDE_CODE_MAX_CONTEXT_TOKENS="$CTX"
# No telemetry/update calls to Anthropic; local generation is slow, so wait long.
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
export API_TIMEOUT_MS="${API_TIMEOUT_MS:-3600000}"
export CLAUDE_CODE_MAX_OUTPUT_TOKENS="${CLAUDE_CODE_MAX_OUTPUT_TOKENS:-4096}"

exec claude "$@"
