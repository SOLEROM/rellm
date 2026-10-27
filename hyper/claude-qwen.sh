#!/usr/bin/env bash
# claude-qwen.sh — run Claude Code against the local HyperQwen server instead of
# Anthropic's API. vLLM serves the Anthropic Messages API at /v1/messages, so no proxy.
#
#   bash claude-qwen.sh                 # interactive
#   bash claude-qwen.sh -p "explain x"  # any claude CLI args pass through
#
# Needs the server running (bash serve.sh) with MAX_LEN >= ~24k: Claude Code's system
# prompt + tool definitions alone are ~15-25k tokens.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PORT:-18020}"
BASE="http://127.0.0.1:$PORT"
MODEL_NAME="${MODEL_NAME:-qwen3.8-27b}"
KEY="${VLLM_API_KEY:-$(cat "$HERE/HyperQwen/api_key.txt" 2>/dev/null || echo local)}"

if [ "$(curl -s -o /dev/null -w '%{http_code}' "$BASE/health")" != 200 ]; then
  echo "HyperQwen server not reachable at $BASE — start it with: bash $HERE/serve.sh" >&2
  exit 1
fi

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
# Real window of the local server (serve.sh MAX_LEN), so Claude Code compacts before
# the server rejects the prompt; also silences the "doesn't recognize the model" warning.
export CLAUDE_CODE_MAX_CONTEXT_TOKENS="${CLAUDE_CODE_MAX_CONTEXT_TOKENS:-28000}"
# No telemetry/update calls to Anthropic; local generation is slow, so wait long.
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
export API_TIMEOUT_MS="${API_TIMEOUT_MS:-3600000}"
export CLAUDE_CODE_MAX_OUTPUT_TOKENS="${CLAUDE_CODE_MAX_OUTPUT_TOKENS:-4096}"

exec claude "$@"
