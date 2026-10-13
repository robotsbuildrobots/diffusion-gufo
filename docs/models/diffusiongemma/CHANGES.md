# What this fork changes in Gufo

A plain-English summary of the work that adds DiffusionGemma to Gufo and makes
it fast on Strix Halo. Measurements and the full record of tried and rejected
ideas are in [BENCHMARKS.md](BENCHMARKS.md), [QUALITY.md](QUALITY.md) and
[EXPERIMENTS.md](EXPERIMENTS.md).

## Adding diffusion

Gufo was built around autoregressive models, which write one token at a time.
DiffusionGemma works differently. It starts from a 256-token "canvas" of random
tokens and refines the whole canvas over several passes ("denoising steps").
When the canvas settles, it is committed and the next canvas starts. Supporting
this needed a new engine and a new server backend.

### The engine (`src/models/diffusion_gemma/`)

- **One set of weights, two jobs.** The model is a Gemma 4 26B-A4B
  mixture-of-experts (30 layers, 128 experts with 8 active per token). The
  engine runs it in two modes:
  - a **causal encoder** that writes the KV cache for the prompt and for each
    accepted canvas, as a normal LLM would;
  - a **bidirectional denoiser**, where every canvas position can see every
    other position as well as the cache.
- **Self-conditioning.** Each denoising step also takes in the model's previous
  guess for the canvas, blended in as a "soft" embedding.
- **The sampler.** It follows the checkpoint's own settings:
  - each step samples a candidate canvas;
  - it keeps the positions the model is most certain of, up to an entropy
    limit;
  - it re-randomises the rest;
  - it stops when the canvas is stable and confident, or after 48 steps.
- **Four weight formats, loaded as they are with no conversion step:**
  - BF16 safetensors;
  - RedHatAI FP8;
  - NVIDIA NVFP4;
  - Unsloth Q8_0 GGUF.

  Strix Halo has no FP8 or FP4 hardware. So the expert weights stay compressed
  in memory and are expanded to BF16 inside the matrix kernels, as they are
  read. The NVFP4 decode is exact.
- **All kernels are Gufo's own, written in HIP:**
  - dense matrix products on WMMA;
  - flash attention for head dimensions 256 (sliding-window layers) and 512
    (global layers);
  - grouped matrix products for the experts;
  - a fused sampler.

  Nothing depends on hipBLASLt, there is no tuning step at startup, and
  repeated runs give bit-identical output.
- **Tests and tooling:**
  - every kernel is tested against float64 references;
  - the tokenizer and chat template are tested;
  - the cache state and sampler decisions are tested;
  - a standalone `gufo-diffusion-gemma` tool runs single prompts, context
    sweeps and reference logits.

### The server (`gufo serve llm`)

- `--model` recognises a DiffusionGemma checkpoint from its `config.json`.
- **APIs and streaming.** Chat Completions, Responses and plain completions all
  work. Streaming sends one canvas (up to 256 tokens) at a time, because that's
  when text becomes final.
- **Thinking.** It can be turned on with `chat_template_kwargs.enable_thinking`
  or `reasoning`, and comes back as `reasoning_content`.
- **Tool calls.** They use the checkpoint's native format and were checked
  byte-for-byte against Transformers on coding-agent conversations. Tested with
  the pi coding agent.
- **Prompt-prefix reuse.** A follow-up request only encodes the part of its
  prompt that is new.
- **Benchmark switches on raw completions:**
  - `ignore_eos` gives fixed-length replies;
  - `cache_prompt: false` disables prefix reuse.
- **JSON-schema output is rejected with a 400 error.** A diffusion sampler
  fills many positions at once, so it can't follow a grammar one token at a
  time.
- **Requests ran one at a time** under the backend's own lock, with no
  batching. Round 2 (below) replaced this with sessions.

## Improving performance

The approach was to measure where each denoising step's time goes, work out
the hardware's floor for each part, and attack the largest gaps. Every change
was measured on its own and then in combination. Several ideas were measured
and rejected.

### What limits the speed

- Generation is limited by memory bandwidth. Each step reads the weights of
  about 77 of the 128 experts in every layer, about 14.6 GB at 8 bits. The
  device reads memory at about 242 GB/s.
- Prefill and long-context attention are limited by compute.

### Changes that were kept

1. **Expert matrix products.**
   - Weights are staged through on-chip shared memory with double buffering.
   - A lighter barrier waits only for shared memory, not for every pending
     memory load.
   - Each wave computes a 2×2 tile.
   - Work is split into 64-row groups per expert, so busy and quiet experts
     balance out.

   They read weights at 180–200 GB/s, about 80% of what the memory system
   delivers.
2. **One dense kernel for every product: canvas, prompt chunks, router, LM
   head.** It replaced hipBLASLt. Weights are reordered once at load into the
   layout the kernel wants. Each product got 20–45% faster, and removing
   hipBLASLt also removed:
   - about 54 s of tuning at startup;
   - 3.3 GB of duplicate weight copies.
3. **Compiler fixes.**
   - Using native vector types for prefetch registers stopped the compiler
     pushing them to slow scratch memory (about 15% on every shape). The same
     fix sped up the BF16 experts by up to 1.8× on prompt chunks.
   - Scheduling barriers stop the compiler hoisting all loads at once and
     spilling (10–25%).
4. **Attention.**
   - Keys are split into up to 16 slices so the small canvas pass fills the
     GPU.
   - The next key/value tile is prefetched while the current one is
     processed. Before this, the kernel was waiting on memory latency, not
     bandwidth.
   - Two head-dimension-512 blocks are allowed to share a compute unit.

   Together these cut a 32K-context global-attention step from 19 ms to
   11.7 ms per layer.
5. **A self-conditioning kernel** that computes the soft embedding directly:
   6–9% per step at short prompts.
6. **Smaller fusions.**
   - The expert combine is folded into the layer-finish kernel.
   - Expert outputs are written in BF16.

### Rejected after measuring

- loading weights directly into matrix fragments;
- BF16-output GEMMs (faster, but they broke agreement with the reference);
- 8-bit dense weights;
- stream-K scheduling;
- fusing the sampler into the LM head;
- a fused-softmax attention;
- larger prefill chunks.

Each rejection is recorded with its numbers in [EXPERIMENTS.md](EXPERIMENTS.md).

### End-to-end effect

FP8, from the last round of changes:

| Context | Prefill | Time per denoising step |
| --- | ---: | ---: |
| 512 | +17% | −11% |
| 32K | +9% | −6% |

BF16 gained more: prefill went up 25–37%.

### Where it ended up (Radeon 8060S)

- **Against llama.cpp's DiffusionGemma (Q8_0):**
  - NVFP4 runs a denoising step 3–4× faster (123 ms vs 393 ms at 2K; 340 vs
    1,365 ms at 121K tokens);
  - it prefills 3.5–7× faster.
- **Against the hardware floor:**
  - a step at 512 tokens of context takes about 134 ms against a floor of about
    95 ms;
  - 32K prefill runs at about 40% of its compute floor, with expert products on
    prompt chunks and global attention as the remaining gap.
- **Against NVIDIA's DGX Spark (GB10) on SPEED-Bench, one request at a time:**
  - 77–84% of GB10's generation speed;
  - 1.2–1.5× longer to the first token.
- **Accuracy:**
  - MMLU: 81.1% with FP8, 80.7% with NVFP4;
  - KL divergence from the Transformers BF16 reference: 0.022 (BF16) to 0.070
    (NVFP4).

## Round 2: faster prefill and concurrent requests

A second pass profiled long prompts in more detail and added a way to serve
several requests at once. Every change was measured on its own and then
together; the numbers are in [BENCHMARKS.md](BENCHMARKS.md#ablation-2026-10-12-round-2).

### Faster prompt processing (prefill)

- **Bigger expert tiles for prompts.** A prompt chunk sends about 128 rows
  to each expert, but the expert kernel worked in 64-row groups, so every
  expert's weights were expanded to BF16 twice. Prompt chunks now use
  128-row groups, run by 16 waves each holding the same small tile as
  before (a first version with bigger tiles per wave ran out of registers).
  Results are bit-identical. Q8_0 keeps 64-row groups, where the bigger
  tile was slower.
- **A rewritten norm and RoPE kernel.** It launched one tiny block per
  head per token, and every block recomputed the same rotary angles with
  `sincosf`, which is slow for the large angles of long prompts. One block
  per token now computes the angles once and shares them across its heads.
  Results are bit-identical.
- **Bigger prompt chunks.** Prompts are now encoded 8192 tokens at a time
  instead of 2048, which keeps the expert groups full. This costs about
  2.4 GB more memory.

Together: prefill is 11–18% faster with NVFP4 and 15–22% faster with FP8,
and time to first token falls with it. A single request's denoising steps
are unchanged.

### Several requests at once (`--sessions N`)

- Each session holds its own KV caches, so up to N requests (several users
  or agents) run at once, each reusing its own prompt history.
- The denoising steps of all running requests are batched into one forward
  pass, and new prompts and finished canvases are encoded in the same pass.
- The gain in total throughput is modest on this GPU: about 19% for short
  question-and-answer requests (MMLU) from four clients, and 10–20% for
  long fixed-length replies. A single 256-token canvas already keeps the
  matrix units busy, so only the expert products get cheaper when canvases
  share a step. With one session, the outputs are bit-identical to before.

### Tried and dropped

- 256-row expert groups for the 8192-token chunks: no faster.
- Two ways of hiding memory latency in the NVFP4 expert kernel: both
  slower. Measurement showed the kernel is limited by fixed work per stage,
  not by decoding FP4 or by reading the weights.
- The global attention kernel, now a third of long-prompt prefill, was left
  as it is: the 64 KB of on-chip memory per workgroup leaves no room for
  bigger tiles at head dimension 512.
