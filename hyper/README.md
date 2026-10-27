# hyper — HyperQwen + Huihui Qwen3.8-27B abliterated

Scripts to install and serve
[huihui-ai/Huihui-Qwen3.8-27B-abliterated](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated)
(W4A16 build: `nchapman/Huihui-Qwen3.8-27B-abliterated-W4A16-GPTQ`) with
[HyperQwen](https://github.com/syv-ai/HyperQwen) (patched vLLM 0.31.0) on any Linux + NVIDIA host,
and to drive it from Claude Code.

## Requirements

| | minimum | good |
|---|---|---|
| GPU | NVIDIA sm75+ (Turing), 8 GB (CPU offload, ~1 tok/s) | 24 GB Ampere+ (~100 tok/s) |
| Driver | supports CUDA 13.0 | |
| Python | `python3.12` + `python3.12-venv` | |
| RAM | 31 GB with an 8 GB GPU (11 GB pinned offload) | 16 GB |
| Disk | ~45 GB free | |
| Tools | `git`, `curl`, `nvidia-smi` | `claude` CLI for `claude-qwen.sh` |

## Quick start

```bash
git clone <this repo> && cd <repo>/hyper
bash install.sh            # ~30-60 min: venv, vLLM, patches, 19.5 GB model, head requant
bash serve.sh              # foreground server on 127.0.0.1:18020 (8+ min to load on small GPUs)
bash test.sh               # other shell: health, models, chat, streaming, tool call
bash claude-qwen.sh        # Claude Code on the local model
```

## Files

| file | what it does |
|---|---|
| `install.sh` | Idempotent install: preflight, clone HyperQwen at the tested commit, venv, patches, download + prepare model, verify. `--no-model` skips the download. |
| `serve.sh` | Picks a profile from the GPU and execs HyperQwen's `single-user/start_qwen.sh`. |
| `test.sh` | 5 smoke tests against the running server; exit 0 = all pass. |
| `claude-qwen.sh` | Runs `claude` against the local server (Anthropic `/v1/messages`). Env vars only, in its own process. |
| `PLAN.md` | Design notes, model choice, hardware limits, measured results. |
| `CLAUDE.md` | Instructions for Claude Code working in this repo. |

## Profiles (`serve.sh`)

- **full**: ≥22 GB VRAM and sm80+. HyperQwen stock path, speculative decoding (`SPEC=mtp`).
- **small**: anything else. fp16, Triton attention, no speculation, `--cpu-offload-gb N`
  with N = 15 + `RESERVE_GB`(4) − VRAM GB, `MAX_LEN=28000`.

Force one with `PROFILE=full|small`. Any HyperQwen knob passes through: `MAX_LEN`, `PORT`,
`VLLM_API_KEY`, `EXTRA_ARGS`, `OFFLOAD_GB`, `RESERVE_GB`, `GPU_UTIL`.

## Install options

| env | default | |
|---|---|---|
| `HQ_DIR` | `./HyperQwen` | where HyperQwen + venv + models live |
| `HQ_REF` | `ce9615c` (tested) | HyperQwen commit; `main` for latest |
| `MODEL_REPO` | `nchapman/Huihui-Qwen3.8-27B-abliterated-W4A16-GPTQ` | HF repo |
| `HQ_EXTRAS` | empty | `"fa2 pthfa kvarn"` optional CUDA builds (sm80+) |
| `SKIP_VERIFY` | 0 | skip `verify.sh` |

## API

OpenAI: `http://127.0.0.1:18020/v1`, model `qwen3.8-27b`. Anthropic: `http://127.0.0.1:18020/v1/messages`.
With no key the server binds 127.0.0.1 only. To expose it, put a key in `HyperQwen/api_key.txt`
(or `VLLM_API_KEY`); never commit it.

## Claude Code

`bash claude-qwen.sh [claude args]` sets `ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN`, maps all model
tiers to `qwen3.8-27b`, and sets `CLAUDE_CODE_MAX_CONTEXT_TOKENS=28000` (raise it to the server's
`MAX_LEN` on a full-profile host). It does not change `~/.claude` or any other running `claude`;
your normal Anthropic-backed `claude` keeps working alongside it.
Expected warnings: "claude.ai connectors are disabled" (only in that process).
On an 8 GB GPU the first turn takes ~19 min (16k-token system prompt prefill).

## Stop

Ctrl-C in the `serve.sh` terminal, or kill that server's PID only.
