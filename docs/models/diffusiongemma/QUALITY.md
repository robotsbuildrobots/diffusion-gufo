# DiffusionGemma quality

Reference: Hugging Face Transformers `DiffusionGemmaForBlockDiffusion`
(transformers 5.17, PyTorch 2.11 ROCm, same checkpoint revision
`f7f5b7f5fa82ffc52addd066915886d497f5517b`), run on the same machine.
Measured 2026-10-10 to 2026-10-12 on Radeon 8060S (`gfx1151`), ROCm 7.2.3.

## Tokenizer and template

`Tokenizer::Encode` is token-for-token identical to `tokenizers` on nine
edge cases (whitespace runs, control bytes, byte fallback, CJK, emoji, RTL,
code, special-token lookalikes) and on a 193,558-token book
(Pride and Prejudice, Project Gutenberg #1342). The cases are kept in
`tests/models/diffusion_gemma/tokenizer_test.cpp`; prompt rendering is
checked in `chat_test.cpp`.

## Denoiser logits

A 2,592-token chat prompt (two encoder chunks, so the sliding-window cache
shift is exercised) followed by two denoiser steps on fixed random
canvases: step 1 without self-conditioning, step 2 self-conditioned on
step 1's logits (temperature 1). KL(reference ‖ x) is per canvas row,
averaged over 256 rows; argmax agreement is the fraction of rows with the
same top token. Random canvases give flat distributions (entropy about 5.8
nats), so argmax agreement is a sensitive, noisy measure.

The noise floor is the spread between valid Transformers configurations of
the same model:

| Step 1 | KL mean | KL p99 | argmax agreement |
| --- | ---: | ---: | ---: |
| Transformers SDPA BF16 vs eager BF16 | 0.0178 | 0.181 | 0.945 |
| Transformers SDPA BF16 vs SDPA FP16 | 0.0174 | 0.138 | 0.926 |
| Transformers eager BF16 vs SDPA FP16 | 0.0117 | 0.045 | 0.980 |

The current build (2026-10-11, evening) against Transformers SDPA BF16, with
the afternoon build (hipBLASLt for prompt chunks) for comparison:

| Gufo weights | Step 1 KL, afternoon → current | Step 2 KL, afternoon → current |
| --- | ---: | ---: |
| FP8 (RedHatAI, `3b3dae46`) | 0.0264 → 0.0293 | 0.0594 → 0.0458 |
| BF16 | 0.0247 → 0.0219 | 0.0309 → 0.0728 |
| Q8_0 (Unsloth GGUF) | 0.0353 → 0.0338 | 0.0596 → 0.0360 |
| NVFP4 (NVIDIA, `ec4ff3df`), current build only | 0.0704 | 0.0860 |

Each of the evening's changes alone, FP8, step 1 / step 2: dense kernel for
the canvas 0.0292 / 0.0701, prompt-chunk kernel 0.0285 / 0.0701, expert
grouping and BF16 expert output 0.0308 / 0.0590. Step 1 stays within 0.003
of the afternoon build for every build; step 2 moves by up to ±0.04 in both
directions (better for FP8 and Q8_0, worse for BF16), which is the
sensitivity described below rather than a consistent shift. The 8-bit
checkpoints against Gufo BF16 (afternoon): FP8 0.0277 / 0.0780, Q8_0
0.0444 / 0.0882.

Round 2 (2026-10-13): with the same 2048-token prompt chunks the round-2
build gives exactly the logits above (every change except the chunk size is
bit-identical). With its default 8192-token chunks the 2,592-token prompt is
one chunk, which changes the order of some attention sums:

| Gufo weights | Step 1 KL, 2048 → 8192 chunks | Step 2 KL, 2048 → 8192 chunks |
| --- | ---: | ---: |
| NVFP4 | 0.0704 → 0.0659 | 0.0860 → 0.0938 |
| FP8 | 0.0292 → 0.0305 | 0.0458 → 0.0682 |
| BF16 | 0.0219 → 0.0166 | 0.0728 → 0.0436 |

Step 1 moves by at most 0.005 and step 2 by up to ±0.03 in both directions,
the same sensitivity as the evening's changes. The round-2 MMLU run below
also uses 8192-token chunks.

NVFP4 diverges about 2.4 times as much as FP8 at step 1. The expert decode is
exact (each 4-bit value times its FP8 block scale is representable in BF16),
and `diffusion_gemma_kernels_hip_test` checks the expert products against a
float64 decode of the same values. The extra divergence is the
checkpoint's 4-bit rounding of the experts, and the MMLU score below shows
how much of it reaches task accuracy.

Step 2 depends chaotically on step 1 (every implementation conditions on its
own step-1 logits; the Transformers variants disagree with each other by KL
0.027–0.056 there). With identical self-conditioning input (Gufo's step-1
logits fed to Transformers), an earlier build's step 2 agreed at KL 0.0158
(p99 0.057), argmax 0.977.

Most of the divergence comes from about ten canvas rows near the start of
the canvas that are very sensitive to rounding: the two Transformers BF16
attention implementations differ by KL 0.15–0.22 on those rows too. Changes
of FP32 summation order alone move the measure within that range:

- The first build measured step-1 KL 0.0154 (BF16). The canvas kernel
  (`dense.hip`) agrees with hipBLASLt per product to about 1e-5 of the
  largest output, yet moves BF16 step 1 to 0.0246; the same build with every
  canvas product sent to hipBLASLt scores 0.0172.
- The hipBLASLt plan chosen by timing changed between starts: two Q8_0
  starts with independently tuned plans differed by KL 0.007 (step 1) and
  0.026 (step 2). Every dense product now runs Gufo's own kernel, so runs
  are bit-identical without a plan database.
- Attention changes (prefetch, two blocks per WGP) are bit-identical.

## Text

Sampled generations are coherent and close to the reference's; for
"Why is the sky blue? Answer in two sentences." Gufo returns *"The sky is
blue because of Rayleigh scattering, where sunlight hits gas molecules in the
atmosphere and scatters light in all directions. Blue light travels in
shorter, smaller waves, which are scattered more than other colors, making
our eyes perceive the sky as blue."*, nearly word-for-word the vLLM FP16
output recorded earlier on this machine.

Long context: a code planted in the middle of 28,985 tokens of
Pride and Prejudice is retrieved exactly ("7341-ORCHID-209").

## API and agent checks

`tests/functional/run.py` (2026-10-11, FP8 weights, `--sessions 1`,
recorded as a baseline): `discovery` and `responses` pass. `stops` passes
every stop-sequence case (buffered and streamed, string, list, overlap,
empty answer, prefix flush) and stops at the thinking case; `tools`,
`structured` and `long-context` need thinking, schema-constrained tool
arguments or JSON response formats, which this backend does not provide
(see README). The runs found and fixed three adapter bugs: a reply cut by
`max_tokens` reported `stop` when an end token followed the limit, a matched
stop sequence ended streams as `cancelled`, and `prefill_tokens` counted the
reused prompt prefix. Thinking and `ignore_eos` were added after this
run; the suites have not been rerun since.

`tests/functional/pi_agent.py` with Pi 0.87.0, one pass: `simple` (twice),
`tools`, `edit`, `creation` and `bugfix` pass, with full prompt-prefix reuse
on every follow-up request. `literal-protocol` writes, reads and verifies the
ChatML-token fixture correctly, but one of its bash calls had a quoting error
that the model corrected on the next call, which the task counts as a
failure.

## MMLU

The full MMLU test (14,042 questions, 57 subjects) through `gufo serve`,
2026-10-12, this branch's build, thinking off. Harness: lm-evaluation-harness
`ad8737ae` task `mmlu_generative`, 5-shot, dataset revision `c30699e8`, the
checkpoint's chat template, the system instruction *"Answer each
multiple-choice question with only the single uppercase letter A, B, C, or D.
Do not include an explanation."*, `max_tokens` 4096, one request at a time.
Exact match after the task's first-line filter.

| Gufo weights | Accuracy | Correct | Answers that are not a single letter | Wall time |
| --- | ---: | ---: | ---: | ---: |
| FP8 (RedHatAI) | 81.1% | 11,388 | 11 | 86 min |
| NVFP4 (NVIDIA) | 80.7% | 11,329 | 12 | 76 min |

The two checkpoints disagree on 681 questions: FP8 alone is right on 370,
NVFP4 alone on 311 (McNemar p ≈ 0.02). The 4-bit experts cost about 0.4
points.

Round 2 (2026-10-13), NVFP4, `--sessions 4` and four concurrent requests,
so that most denoising steps are batched (the build before prompt chunks
and commits joined the batched pass): 80.6% (11,322 correct, 12
answers not a single letter, 69 min). Against the one-at-a-time run the
answers differ in correctness on 21 questions (14 right only one at a
time, 7 only batched; McNemar p ≈ 0.19): batching changes the order of some
floating-point sums, not the accuracy.

These scores are for thinking off and are not comparable with thinking-on
runs. The same harness against the llama.cpp DiffusionGemma server (Q5_K_M
GGUF), whose default chat template enabled thinking (`<|think|>` in the
system turn, about 300 reasoning tokens per question), scored 85.9%. Google
reports MMMLU (multilingual MMLU) at 81.5%, which is a different test.

## Precision choices

- Weights, activations between operations, KV cache and canvas embeddings
  are BF16 as in the checkpoint; norms, softmax, routing, attention
  accumulation and the residual stream are FP32 with BF16 rounding at the
  reference's tensor boundaries.
- Dense products (`kernels::Linear`) produce **FP32 output**; split
  reductions sum FP32 partials in a fixed order. Expert down-projections are
  rounded to BF16, the reference's dtype for that tensor, and combined in
  FP32. hipBLASLt 7.2.3's BF16-output kernels (used by earlier builds) do not round like FP32 accumulation
  followed by round-to-nearest-even; using them doubled the self-conditioned
  KL (0.0118 → 0.031). See [EXPERIMENTS.md](EXPERIMENTS.md).
- The candidate canvas is drawn with Gumbel-max over the temperature-scaled
  logits, distributionally identical to the reference's multinomial draw;
  re-noising uses a host PRNG. Outputs therefore match the reference in
  distribution, not token for token.

## Gaps

- No FP32 ground truth: an FP32 run of this model does not fit in memory, so
  agreement is measured against BF16/FP16 references only.
- MMLU with thinking on, the configuration of the 85.9% llama.cpp run, is
  not measured yet; neither are other task benchmarks.
