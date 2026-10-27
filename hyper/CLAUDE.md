# CLAUDE.md: hyper

Install/serve scripts for local Huihui abliterated Qwen models with two backends:
- `qwen3.6-35b-a3b`: llama.cpp `llama-server` + GGUF (MoE, MTP). Fast on small GPUs. Default below 22 GB.
- `qwen3.8-27b`: HyperQwen (patched vLLM 0.31.0) + W4A16. Fast only on 24 GB Ampere+.

The model table is `models.sh`. `HyperQwen/`, `llama.cpp/`, `models/`, logs and keys are created per
host by the installers and are gitignored; only scripts, templates and docs are versioned.

## Spin it up on a host

1. Check: `nvidia-smi`, `free -g`, `df -h .`; `bash install.sh --list` shows the default for this GPU.
2. `bash install.sh <id> 2>&1 | tee install.log`. Idempotent and resumable; rerun after a failure.
3. Start the server in the background and wait for it:
   `bash serve.sh <id> > serve.log 2>&1 &`, then poll
   `curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:18020/health` until 200
   (~1-4 min for the GGUF, 8+ min for HyperQwen on small GPUs).
4. `bash test.sh`, expecting `6 passed, 0 failed`.
5. Optional: `bash claude-qwen.sh -p "Reply with exactly: HELLO"` (~1.5 min on the GGUF, 8 GB GPU).

To switch models: stop the running server (Ctrl-C or its PID), then `serve.sh <other id>`.

## Rules

- Never edit `~/.claude/` or global Claude settings, and never kill `claude` processes: the user runs
  an Anthropic-backed Claude Code alongside. `claude-qwen.sh` must stay env-var-only.
- Never touch the user's Ollama service or its models.
- Stop only the server you started, by its PID. Never use `pkill -f`: it can match your own shell.
- Do not commit `HyperQwen/`, `llama.cpp/`, `models/`, logs or `api_key.txt` (see `.gitignore`).
- Fix behaviour in these wrapper scripts or `templates/`, not inside `HyperQwen/` or `llama.cpp/`.
  Bump `HQ_REF` / `LLAMA_TAG` deliberately and retest.
- To add a model, add a line to `models.sh`; if its chat template breaks Claude Code, add a
  patched `templates/<id>.jinja` (copy from `curl -s localhost:18020/props | jq -r .chat_template`).
- After changing a script: `bash -n <script>`, then rerun `test.sh` against a live server.

## Known pitfalls

- llama.cpp `--fit on` does nothing if `-ngl` is set ("n_gpu_layers already set by user"); then
  every layer goes to the GPU and loading fails. `serve-gguf.sh` leaves `-ngl` unset unless `CPU_MOE` is set.
- Qwen3.6's stock template raises "System message must be at the beginning" on Claude Code requests;
  `templates/qwen3.6-35b-a3b.jinja` renders later system messages instead.
- HyperQwen on pre-Ampere: no bf16/FA2; `serve.sh` small profile handles it (fp16, TRITON_ATTN, SPEC=off).
- HyperQwen `gptq_marlin_repack ... aten::empty API call failed` at load: too little free VRAM or VMM
  allocator. Raise `RESERVE_GB` or keep `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`.
- Claude Code needs a context >= ~24k; its system prompt is ~16k tokens.
- Models on a network drive (NFS) load slowly the first time (~4 min for 20 GB); later loads use the page cache.
