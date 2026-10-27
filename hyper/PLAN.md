# HyperQwen + Huihui Qwen3.8-27B abliterated — install plan

## Goal

Install [HyperQwen](https://github.com/syv-ai/HyperQwen) (a patched vLLM 0.31.0 serving
stack for Qwen3.8-27B) on this machine, serve
[huihui-ai/Huihui-Qwen3.8-27B-abliterated](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated)
through its OpenAI-compatible API, and test that it works.

## What HyperQwen is

- A pinned vLLM (0.31.0) plus a patch series (`patches/series`, applied by `patches/apply.sh`).
- A model-prep pipeline (`prepare/`) that requantizes `lm_head`, embeddings and the MTP head
  so a 4-bit 27B fits on one 24 GB card.
- Two launchers: `single-user/start_qwen.sh` (speculative decoding, low latency) and
  `batch/start_qwen.sh` (throughput). Both serve on `:18020`, model name `qwen3.8-27b`.
- Two install routes: Docker (`docker compose --profile single up -d`) or a bare-metal venv
  (`docs/install.md`). **This plan uses the venv route** so it is one script.

## Model choice

The Huihui repo is BF16 (27.8B params, ~55 GB) and would need AutoRound/GPTQ quantization
first. Instead we use a ready-made W4A16 build of the same weights:

| candidate | format | fit for HyperQwen |
|---|---|---|
| **nchapman/Huihui-Qwen3.8-27B-abliterated-W4A16-GPTQ** (chosen) | compressed-tensors pack-quantized, 19.5 GB, bf16 heads | Exactly the shape `MODEL=` expects; one 18.6 GB shard → `prepare/quant_heads_stream.py` |
| davetha/…-AutoRound-Int4-baked-embed-int8 | GPTQ layout, built for Intel XPU | Marlin path here does not load AutoRound's GPTQ layout without repacking |
| shawnw3i/…-AWQ-MTP | AWQ | Would work with vLLM, but not the repo's prepared layout |

Override with `MODEL_REPO=<hf-repo> bash install.sh`.

## Hardware reality on this box

| | HyperQwen expects | this machine |
|---|---|---|
| GPU | 24 GB, Ampere+ (sm86 measured) | RTX 2070 Max-Q, **8 GB, Turing sm75** |
| bf16 | yes | **no** (pre-Ampere) → fp16 |
| FlashAttention 2 | yes | **no** on sm75 → Triton attention |
| Speculative decoding | bf16-only kernels | **off** (launcher refuses fp16 + spec) |
| RAM / disk | ≥16 GB / ~40 GB | 31 GB / ~198 GB free ✓ |

The prepared model is ~15 GB of weights; 8 GB of VRAM cannot hold it. The only way to boot
it here is vLLM's `--cpu-offload-gb`, which keeps ~10 GB of weights in pinned RAM and streams
them over PCIe every forward pass. **Expected: roughly 1 token/s.** Turing is "untested"
upstream; this is an experiment. On a 24 GB Ampere+ card the same scripts select the stock
fast path (`PROFILE=full`, ~100+ tok/s per the repo's numbers).

## Steps (what `install.sh` does)

1. **Preflight** — driver, Python 3.12, compute capability (≥7.5 required), VRAM, disk.
2. **Checkout** — reuse `./HyperQwen` (clone at the pinned `HQ_REF` if missing).
3. **venv** — `python3.12 -m venv`, `pip install vllm==0.31.0` + the pinned CUDA 13.0 nvcc
   wheels and flashinfer-cubin from `docs/install.md`.
4. **Patches** — `patches/apply.sh` on the venv's vllm package (skipped if `verify.sh --install` passes).
5. **Optional CUDA builds** — `HQ_EXTRAS="fa2 pthfa kvarn"`; off by default (sm80+ only).
6. **Model** — `prepare/fetch_thirdparty.py <repo>` → `models/<name>`.
7. **Prepare** — `prepare/quant_heads_stream.py` (int8 lm_head/embeddings, int4 MTP, CPU only),
   then `prepare/build_draft_vocab.py` (40k draft head).
8. **Verify** — `verify.sh --install`, then `MODEL=… verify.sh --no-server`.

## Run and test

```bash
bash install.sh          # once (~30-60 min: ~10 GB of wheels + 19.5 GB model + requant)
bash serve.sh            # picks PROFILE=small here, PROFILE=full on a 24 GB Ampere+ card
bash test.sh             # in another shell: health, models, chat, streaming, tool call
```

`serve.sh` small profile = `SPEC=off CTX=fast MAX_LEN=28000 MAX_SEQS=1 VISION=0` plus
`EXTRA_ARGS="--dtype float16 --attention-backend TRITON_ATTN --kv-cache-dtype auto --enforce-eager --cpu-offload-gb N"`,
with N = weights (15) + 4 GB reserve − VRAM, and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`.

Use it from any OpenAI client: base URL `http://127.0.0.1:18020/v1`, model `qwen3.8-27b`,
key from `HyperQwen/api_key.txt` if you create one (no key → binds 127.0.0.1 only).

## Risks

- **Turing kernels**: Marlin W4A16 supports sm75; the GDN (linear-attention) Triton kernels and
  fp16 activations on this model are unverified. Qwen in fp16 can overflow → garbage text.
- **Speed**: offload makes it usable for smoke tests, not for real work.
- **RAM**: ~10 GB pinned + loader peak; close other heavy apps before booting.

## Fallback if vLLM cannot run it here

Serve the official GGUF (`huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF`, Q4_K_M) with
llama.cpp's `llama-server` and partial GPU offload — same OpenAI API, ~2-4 tok/s on this card.
That is outside HyperQwen, so it is not scripted here.

## Results (2026-10-27, this machine)

**It installs and serves.** RTX 2070 Max-Q 8 GB, driver 595.99, vLLM 0.31.0 + all HyperQwen patches.

| step | result |
|---|---|
| `install.sh` | venv + vLLM 0.31.0, full patch series applied, model downloaded + heads requantized + draft vocab built |
| `verify.sh --install` | all patches PASS (KVarN not installed: optional) |
| `verify.sh --no-server` (model) | 7/7 model checks PASS |
| boot #1 (`OFFLOAD_GB=9`, VMM allocator) | **failed** in `gptq_marlin_repack` (`aten::empty ... API call failed`) |
| boot #2 (`OFFLOAD_GB=11`, `expandable_segments:False`) | **up**: 3.25 GiB on GPU, 11 GiB in pinned RAM, 2.89 GiB KV (28,431 tokens); ~8 min to ready |
| `test.sh` | **5/5 PASS**: health, models, chat ("Paris"), SSE streaming, tool call |
| speed | prefill ~33 tok/s, decode **~0.8 tok/s** |

Kernels picked on sm75: MarlinLinearKernel (W4A16), TRITON_ATTN, Triton/FLA GDN.
`serve.sh` now defaults to the boot #2 settings (`RESERVE_GB=4` → 11 GB offload, VMM off).

Verdict: functional for testing and API integration, too slow for interactive use.
For real use, run the same `install.sh` + `serve.sh` on a 24 GB Ampere+ GPU
(`PROFILE=full`, speculative decoding, ~100+ tok/s per upstream numbers), or use the
llama.cpp GGUF fallback above.

## Claude Code on the local model

vLLM serves the Anthropic Messages API (`/v1/messages`), so Claude Code talks to it directly:
`bash claude-qwen.sh` sets `ANTHROPIC_BASE_URL=http://127.0.0.1:18020`, maps every model tier
to `qwen3.8-27b`, and runs `claude`. Needs `MAX_LEN` >= ~24k (serve.sh small default is now 28000).

Tested 2026-10-27 on the 8 GB card: `claude-qwen.sh -p "Reply with exactly the word: HELLO"` →
`HELLO`, success, ~16.6k-token prompt, **19 min to first token** (prefill), then ~0.8 tok/s.
The prefix cache makes later turns of the same session faster (system prompt reused).
The `total_cost_usd` Claude Code prints is its own estimate; nothing is billed.
