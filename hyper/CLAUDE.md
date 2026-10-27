# CLAUDE.md — hyper

Install/serve scripts for HyperQwen (patched vLLM 0.31.0) running Huihui Qwen3.8-27B abliterated
(W4A16). The upstream repo is cloned into `./HyperQwen` by `install.sh` and is gitignored; only the
wrapper scripts and docs here are versioned.

## Spin it up on a host

1. Check: `nvidia-smi` (sm75+), `python3.12 -c 'import venv'`, `df -h .` (≥45 GB).
2. `bash install.sh 2>&1 | tee install.log`. Idempotent; rerun after a failure.
3. Start the server in the background and wait for it:
   `bash serve.sh > serve.log 2>&1 &` then poll `curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:18020/health` until 200
   (look for `Application startup complete` in serve.log; 8+ min on small GPUs).
4. `bash test.sh`, expecting `5 passed, 0 failed`.
5. Optional: `bash claude-qwen.sh -p "Reply with exactly: HELLO"`.

## Rules

- Never edit `~/.claude/` or global Claude settings, and never kill `claude` processes: the user runs
  an Anthropic-backed Claude Code alongside. `claude-qwen.sh` must stay env-var-only.
- Stop only the server you started, by its PID. Never use `pkill -f` (it can match your own shell).
- Do not commit `HyperQwen/`, models, logs or `api_key.txt` (see `.gitignore`).
- Fix behaviour in these wrapper scripts, not by editing files inside `HyperQwen/`; bump `HQ_REF` in
  `install.sh` deliberately and retest.
- After changing a script: `bash -n <script>`, then rerun `test.sh` against a live server.

## Known pitfalls (from PLAN.md)

- Pre-Ampere: no bf16/FA2. `serve.sh` small profile handles it (fp16, TRITON_ATTN, SPEC=off).
- `gptq_marlin_repack ... aten::empty API call failed` at load: too little free VRAM or VMM allocator.
  Raise `RESERVE_GB` or keep `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`.
- Claude Code needs `MAX_LEN` ≥ ~24k; its system prompt is ~16k tokens.
- `install.sh` checks heads with `"lm_head.weight_packed"` in the safetensors index; the requant
  leaves `*.bak-orig` files (~15 GB) that can be deleted once it is verified.
