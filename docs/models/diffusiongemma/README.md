# DiffusionGemma 26B-A4B

Native HIP block-diffusion text generation on AMD Strix Halo (`gfx1151`) for
[`google/diffusiongemma-26B-A4B-it`](https://huggingface.co/google/diffusiongemma-26B-A4B-it).
Loads the original BF16 safetensors, the NVIDIA NVFP4 checkpoint, the RedHatAI
FP8 checkpoint or a Q8_0 GGUF directly (no conversion); Python is not needed
for inference.

DiffusionGemma is a Gemma 4 26B-A4B mixture-of-experts model (30 layers,
128 experts with 8 active, sliding-window and global attention) trained for
block diffusion. One set of layer weights serves two roles:

- a **causal encoder** that writes the KV cache for the prompt and for each
  accepted canvas, and
- a **bidirectional denoising decoder** that refines a 256-token canvas,
  attending to the cache plus the whole canvas, with self-conditioning on
  its previous prediction.

Each canvas starts as uniform random tokens. Every denoising step samples a
candidate canvas, accepts the lowest-entropy positions under an entropy
bound, re-noises the rest, and stops when the argmax canvas is stable and
confident (or after 48 steps). The accepted argmax canvas is appended and
the next canvas begins. Defaults come from the checkpoint's
`generation_config.json`.

## Weights

About 50 GB (BF16, 11 shards). The engine reads the text tensors only
(about 47 GiB resident); the vision tower is not loaded.

```sh
hf download google/diffusiongemma-26B-A4B-it \
  --revision f7f5b7f5fa82ffc52addd066915886d497f5517b \
  --local-dir models/diffusiongemma
```

Quantized alternatives keep the experts quantized on the GPU and widen them
to BF16 inside the expert kernels; the other matrices are dequantized to BF16
at load. The 8-bit checkpoints need about 27 GiB resident and run at the
same speed per step; NVFP4 is the fastest, at 13% less time per step than FP8
at 512 tokens of context, because generation is limited by reading expert weights:

| Checkpoint | Format | Mean KL to BF16 reference |
| --- | --- | ---: |
| [`RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic`](https://huggingface.co/RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic) (revision `3b3dae46`) | FP8 E4M3, one scale per output channel (weights only; activations stay BF16) | 0.029 |
| Unsloth `diffusiongemma-26B-A4B-it-Q8_0.gguf`, in a directory with the BF16 checkpoint's `config.json` and tokenizer files | Q8_0 | 0.034 |
| [`nvidia/diffusiongemma-26B-A4B-it-NVFP4`](https://huggingface.co/nvidia/diffusiongemma-26B-A4B-it-NVFP4) (revision `ec4ff3df`) | NVFP4 experts (E2M1 values, an FP8 scale per 16 and an FP32 scale per matrix; decoded exactly to BF16, activations stay BF16); other matrices BF16 | 0.070 |

The KL column is the first denoising step's mean per-position divergence on a
fixed prompt and canvas from the Transformers BF16 forward; Gufo BF16 itself
scores 0.022 there, and two Transformers attention implementations differ by
0.018 ([QUALITY.md](QUALITY.md)).

The BF16 weights need about 47 GiB of GPU-visible memory (GTT on Strix Halo) plus
1.4 GiB for a blocked copy of the LM head, about 1 GiB of scratch and 20 KiB
per context token for the global-layer KV
cache. Stop other GPU servers first.

## Serve

```sh
gufo serve --port 8080 llm --model models/diffusiongemma \
  --served-model-name diffusiongemma --context 131072
```

`--model` takes the checkpoint directory; it is recognised by its
`config.json` (`model_type: diffusion_gemma`). Startup loads the weights
(about 5 s from a warm page cache), blocks the dense matrices in place for
the matrix kernel and runs one prompt chunk and two denoising steps, so the
first request runs at full speed. Every kernel is Gufo's own: there is no
tuning step, and repeated runs are bit-identical.

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"diffusiongemma","messages":[{"role":"user","content":"Why is the sky blue?"}]}'
```

| Behavior | Notes |
| --- | --- |
| Output | Chat Completions, Responses and completions; streaming delivers one canvas (up to 256 tokens) at a time. |
| Sampling | The diffusion sampler replaces token sampling: `temperature`, `top_k`, `top_p` and penalties do not apply. `seed` makes a request reproducible within a build. |
| Length | `max_tokens` (default: until EOS or the context is full); generation runs whole canvases and the reply is truncated to the limit (finish reason `length` unless an end token falls within it). On `/v1/completions`, `ignore_eos: true` never samples the end tokens, so the reply runs to `max_tokens` (fixed-length benchmarks). |
| Thinking | Off by default; a thinking channel the model emits is removed. `chat_template_kwargs.enable_thinking: true` (or `reasoning`) renders the prompt with thinking on and returns the channel as `reasoning_content`. Raw completions keep it as `<think>...</think>` text. |
| Tools | Function tools in the checkpoint's native format (declarations, calls and results rendered as `chat_template.jinja` does, checked byte-for-byte against Transformers on coding-agent conversations); calls are returned as OpenAI `tool_calls`, streamed once each call is complete. Tested with the pi coding agent. |
| Prompt reuse | A request reuses the longest prompt prefix it shares with the previous request's cache (its prompt and committed reply canvases) when the divergence lies within the last 4096 cached tokens; only the rest is encoded. Logged as `reused=` and reported as cached prompt tokens. |
| Structured output | Not supported: `response_format` and Responses `text.format` are rejected with 400 `unsupported_response_format`, because the diffusion sampler cannot apply a JSON grammar token by token. For the same reason tool arguments are not constrained to their schema. |
| Not supported | Images, concurrent sessions (requests run one at a time). |

## Command-line tool

`gufo-diffusion-gemma` (built beside `gufo`) checks the tokenizer and chat
template, runs single prompts, writes reference logits for
[QUALITY.md](QUALITY.md) and runs the context sweep in
[BENCHMARKS.md](BENCHMARKS.md):

```sh
gufo-diffusion-gemma generate --model models/diffusiongemma \
  --prompt "Write a haiku about lighthouses." --max-tokens 256
gufo-diffusion-gemma bench --model models/diffusiongemma --corpus book.txt \
  --contexts 512,2048,8192,32768 --reps 3 --gen-tokens 512
```

`--profile` prints per-phase GPU time for each request.

Tests: `diffusion_gemma_kernels_hip_test` checks every kernel family against
float64 references (GPU), including the NVFP4 decode and the sampler's
end-token suppression; `diffusion_gemma_cache_state_test` and
`diffusion_gemma_denoise_test` cover prompt reuse, cache slots and the
sampler's decisions (CPU).

## Implementation

`src/models/diffusion_gemma/`:

| File | Contents |
| --- | --- |
| `config.*` | `config.json` / `generation_config.json`, checked against the kernel shapes. |
| `tokenizer.*`, `chat.*` | Gemma 4 BPE tokenizer (byte fallback) and the chat template for text conversations. |
| `weights.*` | Safetensors index, fused QKV / gate-up placement, BF16 loader. |
| `weights_gguf.cpp`, `weights_fp8.cpp`, `weights_nvfp4.cpp` | Q8_0 GGUF, FP8 and NVFP4 checkpoint loaders: quantized experts kept on the device as stored, other matrices dequantized to BF16. |
| `upload.*` | Checkpoint ranges to device memory through Gufo's shared weight upload (`src/core/hip/weight_upload`), with batched device staging for tensors converted on load. |
| `engine.*` | Encoder and decoder passes, the block-diffusion loop, prompt reuse. |
| `cache_state.*` | KV-cache bookkeeping: global and sliding slots, reuse history, prefix reuse. |
| `denoise.*` | The sampler's host decisions: temperature schedule, entropy-bound acceptance, convergence, reply length. |
| `dense.hip` | Every dense product (QKV, output, MLP, router, self-conditioning MLP, LM head) for any row count: FP32 output, 256 x 128 (or 128 x 128) tiles over row-major activations and weights blocked in place at load, 64-deep stages with register prefetch; ordered split reduction for outputs with few tiles. |
| `soft_embed.hip` | Self-conditioning soft embedding over the whole vocabulary: all 256 canvas rows per workgroup (the embedding table is read once), vocabulary split with ordered FP32 partial sums, probabilities in a blocked layout written by the sampler. |
| `attention.hip` | WMMA flash attention for head dimensions 256 and 512: causal + sliding window (encoder) and bidirectional (decoder), split-key with merge for short query sets. The next key/value tile is loaded into registers while the current one is computed; head dimension 512 runs two blocks per WGP. |
| `moe.hip` | Grouped expert GEMMs (BF16, Q8_0, FP8 or NVFP4 weights; NVFP4 is decoded with a byte-permute table built from each block scale), one workgroup per 64-row group of an expert: gate/up with fused GELU, down with BF16 output; LDS-staged and double-buffered with LDS-only barriers. |
| `kernels.hip` | Norms, RoPE (default and proportional), routing, dispatch and its group list, the layer finish with the expert combine, the per-row sampler (softcap, temperature, end-token suppression, entropy, argmax, Gumbel sample, self-conditioning softmax), dequantization. |
| `device.hpp`, `hip_check.hpp`, `profiler.*` | WMMA fragment helpers and layout, HIP error checks, per-phase GPU timing. |

The HTTP adapter is `src/cli/serve/diffusion_gemma_backend.*`.
