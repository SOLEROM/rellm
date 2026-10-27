#!/usr/bin/env bash
# test.sh — smoke-test a running HyperQwen server: health, model list, a chat
# completion, streaming, and a tool call. Exit 0 only if every check passes.
#
#   bash test.sh                    # against http://127.0.0.1:18020
#   PORT=18020 TIMEOUT=900 bash test.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HQ_DIR="${HQ_DIR:-$HERE/HyperQwen}"
BASE="http://127.0.0.1:${PORT:-18020}"
TIMEOUT="${TIMEOUT:-600}"          # per request; offloaded 27B is slow
KEY="${VLLM_API_KEY:-$(cat "$HQ_DIR/api_key.txt" 2>/dev/null || true)}"
AUTH=(); [ -n "$KEY" ] && AUTH=(-H "Authorization: Bearer $KEY")
PASS=0; FAIL=0

ok()   { echo "PASS  $1"; PASS=$((PASS+1)); }
bad()  { echo "FAIL  $1"; FAIL=$((FAIL+1)); }
post() { curl -s --max-time "$TIMEOUT" "${AUTH[@]}" -H 'Content-Type: application/json' "$BASE$1" -d "$2"; }

[ "$(curl -s -o /dev/null -w '%{http_code}' "$BASE/health")" = 200 ] && ok "/health" || { bad "/health (server not up?)"; exit 1; }

MODELS=$(curl -s "${AUTH[@]}" "$BASE/v1/models")
echo "$MODELS" | grep -q '"qwen3.8-27b"' && ok "/v1/models lists qwen3.8-27b" || bad "/v1/models: $MODELS"

T0=$(date +%s.%N)
R=$(post /v1/chat/completions '{"model":"qwen3.8-27b","max_tokens":48,"temperature":0,
  "messages":[{"role":"user","content":"What is the capital of France? Answer in one word."}],
  "chat_template_kwargs":{"enable_thinking":false}}')
T1=$(date +%s.%N)
ANS=$(echo "$R" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d["choices"][0]["message"]["content"])' 2>/dev/null)
TOK=$(echo "$R" | python3 -c 'import json,sys;print(json.load(sys.stdin)["usage"]["completion_tokens"])' 2>/dev/null || echo 0)
echo "      answer: ${ANS:-<none>}   ($TOK tokens, $(python3 -c "print(f'{$TOK/($T1-$T0):.2f}')" 2>/dev/null) tok/s e2e)"
echo "$ANS" | grep -qi paris && ok "chat completion is correct" || bad "chat completion: $R"

S=$(post /v1/chat/completions '{"model":"qwen3.8-27b","max_tokens":16,"stream":true,
  "messages":[{"role":"user","content":"Count: 1, 2, 3,"}],"chat_template_kwargs":{"enable_thinking":false}}')
echo "$S" | grep -q 'data: \[DONE\]' && ok "streaming (SSE) completes" || bad "streaming: ${S:0:300}"

TC=$(post /v1/chat/completions '{"model":"qwen3.8-27b","max_tokens":128,"temperature":0,"tool_choice":"auto",
  "messages":[{"role":"user","content":"What is the weather in Berlin? Use the tool."}],
  "chat_template_kwargs":{"enable_thinking":false},
  "tools":[{"type":"function","function":{"name":"get_weather","description":"Get weather for a city",
    "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]}')
echo "$TC" | grep -q '"get_weather"' && ok "tool call emitted" || bad "tool call: ${TC:0:400}"

echo; echo "$PASS passed, $FAIL failed"
[ "$FAIL" = 0 ]
