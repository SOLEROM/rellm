# hyper: local uncensored Qwen models behind an OpenAI + Anthropic API

Install and run Huihui abliterated Qwen models on any Linux + NVIDIA host, and drive them from
Claude Code. One spin script, two backends:

| model id | backend | size | 8 GB RTX 2070 (measured) | 24 GB Ampere+ |
|---|---|---|---|---|
| `qwen3.6-35b-a3b` | llama.cpp (GGUF Q4_K, MoE, 3B active, MTP) | 20 GB | **~17 tok/s**, prompt ~280 tok/s | faster |
| `qwen3.8-27b` | [HyperQwen](https://github.com/syv-ai/HyperQwen) (patched vLLM 0.31.0, W4A16) | 19.5 GB | ~0.8 tok/s, prompt ~33 tok/s | ~100+ tok/s |

Models: [huihui-ai/Huihui-Qwen3.6-35B-A3B-abliterated-MTP-GGUF](https://huggingface.co/huihui-ai/Huihui-Qwen3.6-35B-A3B-abliterated-MTP-GGUF),
[huihui-ai/Huihui-Qwen3.8-27B-abliterated](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated)
(served from the W4A16 build `nchapman/Huihui-Qwen3.8-27B-abliterated-W4A16-GPTQ`).
With no model named, the scripts pick `qwen3.8-27b` on a ≥22 GB sm80+ GPU and `qwen3.6-35b-a3b` elsewhere.

## Quick start

```bash
git clone <this repo> && cd <repo>/hyper
bash install.sh --list                # models
bash install.sh qwen3.6-35b-a3b       # or qwen3.8-27b; no arg = default for this GPU
bash serve.sh   qwen3.6-35b-a3b       # foreground on 127.0.0.1:18020
bash test.sh                          # other shell: 6 smoke tests
bash claude-qwen.sh                   # Claude Code on whichever model is serving
```

One model serves at a time (same port, same GPU). Stop it with Ctrl-C, then start another.

## Requirements

| | `qwen3.6-35b-a3b` (llama.cpp) | `qwen3.8-27b` (HyperQwen) |
|---|---|---|
| GPU | NVIDIA, any recent driver (CPU-only works, slowly) | sm75+, best 24 GB sm80+ |
| RAM | 32 GB with an 8 GB GPU (experts live in RAM) | 31 GB with 8 GB GPU, else 16 GB |
| Disk | ~22 GB | ~45 GB (venv + model + requant backups) |
| Software | `curl`, `tar`, `python3` | `python3.12` + venv, `git` |
| Install time | ~10 min (download) | ~30-60 min |

`claude-qwen.sh` also needs the `claude` CLI.

## Files

| file | what it does |
|---|---|
| `models.sh` | Model table (id, backend, HF repo, file). Add a line to add a model. |
| `install.sh` | `install.sh <id>`: HyperQwen flow here; GGUF models go to `install-gguf.sh`. Idempotent. |
| `install-gguf.sh` | Prebuilt llama.cpp release (pinned `b11934`, CUDA 12.8) + resumable GGUF download + sha256 check. |
| `serve.sh` | `serve.sh <id>`: HyperQwen launcher with GPU profiles; GGUF models go to `serve-gguf.sh`. |
| `serve-gguf.sh` | `llama-server` with auto GPU/RAM fit, MTP speculative decoding, 64k context. |
| `templates/` | Patched chat templates (`<id>.jinja`), used automatically by `serve-gguf.sh`. |
| `test.sh` | Health, models, chat, streaming, tool call, Anthropic `/v1/messages`. |
| `claude-qwen.sh` | Claude Code on the local server; reads model name + context from it. |
| `PLAN.md` | Design notes and measured results. `CLAUDE.md`: instructions for Claude Code here. |

## Knobs

`serve-gguf.sh`: `CTX_SIZE` (65536), `SPEC=mtp|off`, `MTP_N` (6), `CPU_MOE=N` (fixed number of MoE
layers in RAM instead of auto-fit), `HOST`, `PORT`, `EXTRA_ARGS`.
`install-gguf.sh`: `LLAMA_TAG`, `LLAMA_FLAVOR=cuda-12.8|cuda-13.4|vulkan|cpu`, `GGUF_FILE` (another
quant from the same repo, e.g. `...-Q3_K.gguf` for 16 GB RAM hosts), `GGUF_DIR`, `LLAMA_DIR`.
`serve.sh` (HyperQwen): `PROFILE=full|small`, `MAX_LEN`, `OFFLOAD_GB`, `RESERVE_GB`, `GPU_UTIL`,
`EXTRA_ARGS`. `install.sh` (HyperQwen): `HQ_REF` (pinned `ce9615c`), `HQ_DIR`, `MODEL_REPO`, `HQ_EXTRAS`.

## API

OpenAI: `http://127.0.0.1:18020/v1`, Anthropic: `http://127.0.0.1:18020/v1/messages`.
The model name is the model id (`qwen3.6-35b-a3b` / `qwen3.8-27b`).
Without a key the server binds 127.0.0.1 only. To expose it, put a key in `api_key.txt`
(or `VLLM_API_KEY`) and set `HOST=0.0.0.0`; never commit the key (it is gitignored).

## Claude Code

`bash claude-qwen.sh [claude args]` points `ANTHROPIC_BASE_URL` at the local server, maps every model
tier to the served model, and sets `CLAUDE_CODE_MAX_CONTEXT_TOKENS` from the server's context size.
It only sets env vars for that one process: `~/.claude`, your login, and other running `claude`
sessions are untouched, so the normal Anthropic-backed `claude` keeps working alongside.
The "claude.ai connectors are disabled" warning is expected (that process only).
First turn (~16k-token system prompt): ~1 min on `qwen3.6-35b-a3b`, ~19 min on `qwen3.8-27b` with 8 GB.
