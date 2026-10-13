# DiffusionGemma benchmarks

Measured 2026-10-11 and 2026-10-12 on a Ryzen AI Max+ 395 / Radeon 8060S
(`gfx1151`, 121 GiB unified memory, measured device read bandwidth 242 GB/s),
Linux 7.2.5, ROCm 7.2.3 (Ubuntu 24.04 container), Gufo `92aaed5` plus this
DiffusionGemma port. FP8 is the main format. One request at a time (except
under "Concurrent requests"); no other GPU workload.

Weights: `google/diffusiongemma-26B-A4B-it` revision `f7f5b7f5` (BF16),
`RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic` revision `3b3dae46` (FP8),
`nvidia/diffusiongemma-26B-A4B-it-NVFP4` revision `ec4ff3df` (NVFP4) and
Unsloth `diffusiongemma-26B-A4B-it-Q8_0.gguf` (Q8_0). Reference:
llama.cpp's DiffusionGemma server with the same Q8_0 GGUF (`-ngl 99
--flash-attn`), measured 2026-10-10.

Server: `gufo serve llm --context 131072`. Before each measured request a short unrelated request
replaces the previous prompt in the cache, so no prompt reuse is measured.
Every number below is from the server's own log line for the request.

## Short prompt (PG19 test)

A 508-token prompt asking to continue a passage from the PG19 set, streamed
Chat Completions, `max_tokens` 512, seeds fixed; mean of 3 runs.

| Setup | Time to first token s | Tokens / s (whole request) | ms / step | Steps | Reply tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gufo NVFP4 | 2.70 | 81.8 | 115.5 | 52 | 512 |
| Gufo FP8 | 4.37 | 63.7 | 135.7 | 57 | 512 |
| Gufo Q8_0 | 3.94 | 60.9 | 135.0 | 60 | 512 |
| Gufo BF16 | 5.20 | 52.7 | 182.4 | 51 | 512 |
| llama.cpp Q8_0 | 12.34 | 20.7 | 386 | 64 | 512 |

Tokens per second depends on the engine's time per step and on how many
tokens each step completes; the second depends on the text and the weights.
Compare engines by ms / step. llama.cpp has no FP8 tensor type; its row uses
the Q8_0 GGUF.

## Long prompts

The first *N* tokens of a book in one user turn, 512 tokens out, one
request per length (two for 2K–32K, which agree within 1%).

| Prompt tokens | Prefill tok/s NVFP4 | FP8 | Q8_0 | BF16 | llama.cpp Q8_0 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2,261 | 3,443 | 3,116 | 3,180 | 2,901 | 978 |
| 8,500 | 3,207 | 2,939 | 2,995 | 2,912 | 713 |
| 33,105 | 2,537 | 2,326 | 2,363 | 2,387 | 445 |
| 65,389 | 1,957 | 1,831 | 1,853 | 1,866 | 310 |
| 121,294 | 1,379 | 1,329 | 1,339 | 1,344 | 201 |

| Prompt tokens | Time to first token s NVFP4 | FP8 | Q8_0 | BF16 | llama.cpp Q8_0 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2,261 | 5.5 | 5.9 | 6.0 | 7.0 | 18.8 |
| 8,500 | 8.7 | 9.3 | 8.7 | 9.5 | 27.2 |
| 33,105 | 18.2 | 21.2 | 20.8 | 21.8 | 67.9 |
| 65,389 | 44.7 | 47.0 | 45.8 | 47.1 | 142.7 |
| 121,294 | 104.7 | 108.4 | 107.8 | 106.3 | 368.3 |

The first visible tokens arrive with the first 256-token canvas: prefill
plus one block of denoising.

| Prompt tokens | ms / step NVFP4 | FP8 | Q8_0 | BF16 | llama.cpp Q8_0 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2,261 | 123 | 141 | 141 | 184 | 393 |
| 8,500 | 134 | 152 | 153 | 194 | 441 |
| 33,105 | 176 | 198 | 197 | 237 | 646 |
| 65,389 | 243 | 262 | 261 | 293 | 937 |
| 121,294 | 340 | 350 | 361 | 396 | 1,365 |

Startup to the first accepted request: 8–10 s (load 7–9 s, warmup 1.1–1.2
s; NVFP4 6 s); there is no tuning step. The NVFP4 rows were measured
2026-10-12 with the same protocol.

## SPEED-Bench against DGX Spark (GB10)

[diffu-benching](https://github.com/iv2nl0b9v/diffu-benching) published
DiffusionGemma NVFP4 results for vLLM on an NVIDIA GB10 (DGX Spark). The same
workload on Gufo, measured 2026-10-12 (round 1) and 2026-10-13 (round 2):
SPEED-Bench `throughput_2k`/`8k`/`16k`/`32k`, their recorded prompts rendered
with the checkpoint's chat template (thinking on; token-identical to the
Hugging Face tokenizer vLLM uses: 2,274 / 9,088 / 17,815 / 36,352 tokens on
average), 20 requests after 2 warm-ups, sequential sampling, seed 42, end
tokens suppressed so that every reply is exactly 1,024 tokens, AIPerf
0.12.0, `/v1/completions` with `return_token_ids` and `cache_prompt: false`,
`--context 65536`. Every run is scored with diffu-benching's own audit code
and passes it (1,024 tokens, no stop tokens, 256-token first chunk) and a
check that the streamed text equals the decoded token IDs. GB10 figures are
theirs: the released (non-rolling) sampler, canvas 256, up to 48 steps
(`results/rolling-matrix` and `results/rolling-long-context`).

Generation tok/s is their corrected per-user rate after the first canvas;
total tok/s is output over the whole run; TTFT is seconds and, at
concurrency 2 and 4, includes waiting for a session (round 1 served one
request at a time).

Concurrency 1:

| Split | GB10 gen tok/s | Gufo round 1 | Gufo round 2 | GB10 TTFT | Gufo round 1 | Gufo round 2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2K | 142.5 | 120.2 | 115.6 | 2.21 | 2.63 | 2.71 |
| 8K | 130.3 | 100.0 | 100.1 | 3.50 | 5.17 | 5.07 |
| 16K | 121.0 | 94.1 | 98.3 | 6.51 | 9.49 | 8.37 |
| 32K | 92.9 | 73.3 | 76.2 | 13.55 | 19.61 | 17.74 |

Prefill from the server's log, round 1 → round 2: 3,416 → 4,018 /
3,143 → 3,667 / 2,867 → 3,306 / 2,406 → 2,688 tok/s (+12 to +18%); ms per
step unchanged. The time to first token also holds the first canvas's
denoising, whose step count varies from run to run, so at 2K and 8K the
faster prefill is within that spread; at 16K and 32K it is 10-12% sooner.

Several requests at once (Gufo round 2 with `--sessions 4`):

| Split | Concurrency | GB10 total tok/s | Gufo round 1 | Gufo round 2 | GB10 TTFT | Gufo round 1 | Gufo round 2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2K | 1 | 127.5 | 104.5 | 101.9 | 2.21 | 2.63 | 2.71 |
| 2K | 2 | 170.6 | 102.4 | 106.7 | 3.27 | 12.23 | 4.57 |
| 2K | 4 | 206.5 | 97.9 | 126.5 | 5.20 | 30.51 | 8.66 |
| 8K | 1 | 102.8 | 75.4 | 74.8 | 3.50 | 5.17 | 5.07 |
| 8K | 2 | 130.9 | 72.4 | 82.6 | 5.18 | 18.75 | 8.20 |
| 8K | 4 | 149.9 | 74.9 | 88.2 | 8.64 | 42.23 | 14.68 |
| 16K | 1 | 74.9 | 53.8 | 57.3 | 6.51 | 9.49 | 8.37 |
| 16K | 2 | 87.0 | 55.3 | 61.3 | 10.43 | 27.16 | 14.76 |
| 16K | 4 | 95.2 | 55.1 | 63.9 | 16.01 | 60.04 | 26.59 |
| 32K | 1 | 45.1 | 32.0 | 34.2 | 13.55 | 19.61 | 17.74 |
| 32K | 2 | 49.8 | 32.5 | 35.5 | 18.67 | 49.20 | 27.91 |
| 32K | 4 | 53.4 | 34.0 | 36.7 | 33.73 | 98.97 | 54.10 |

At concurrency 4, round 2 gives 8-29% more total output than round 1 and
first tokens 1.8-3.5 times sooner, and Strix Halo reaches 59-69% of GB10's
total throughput (round 1: 47-64%). GB10 gains more from batching, probably
because its matrix units (about twice as fast) are not saturated by one
canvas, while Strix Halo's are. FP8 on Strix Halo, concurrency 1, round 1, generation tok/s: 97.1 /
88.5 / 85.8 / 71.4.

## Literary continuation (spark-gemma-lab PG19)

The profiling workload of
[spark-gemma-lab](https://github.com/bodono/spark-gemma-lab) (`f6e292f`),
measured 2026-10-13 with the round-2 build: 512-token excerpts of PG19
test-set books (`prepare_pg19.py --count 96 --input-tokens 512`, seed 42,
569 tokens after chat formatting), each wrapped in the lab's instruction to
continue the passage in the same style, thinking off, natural end of text,
`max_tokens` 512, 20 measured requests after one warm-up, one request at a
time (`--sessions 1`, `--context 8192`). Requests and metrics come from the
lab's own runner (`server/runner.mjs`, `server/metrics.mjs`); the one change
to its request is dropping `ignore_eos: false`, which Gufo rejects on chat.
Steps per canvas and ms per step are from the server's log.

| Weights | Steps per canvas | ms / step | tok/s after the first canvas | tok/s per request | Time to first token s |
| --- | ---: | ---: | ---: | ---: | ---: |
| NVFP4 | 32.5 | 116.9 | 65.4 | 66.1 | 3.95 |
| FP8 | 30.8 | 138.0 | 59.9 | 59.1 | 4.53 |

Every reply ran the full 512 tokens (two canvases). A step costs the same as
on SPEED-Bench, but a canvas of narrative prose takes about 32 denoising
steps to settle against about 19 on SPEED-Bench (round 2, concurrency 1, all
lengths), so generation is about 43% slower (65 against 116 tok/s at 2K).
The model is less certain about open-ended prose, and the entropy bound
accepts fewer positions per step. Time to first token is almost all
first-canvas denoising; the 569-token prompt prefills in about 0.2 s. Only
sampler changes (fewer steps per canvas) can close this gap. The lab
publishes no GB10 results for this workload.

## Ablation (2026-10-12, round 2)

Each change alone on the round-1 build, then all retained changes, with
`gufo-diffusion-gemma bench` (story continuation, 512 tokens out, mean of
two interleaved rounds). Prefill tok/s and ms per denoising step, change
against the baseline. Details in [EXPERIMENTS.md](EXPERIMENTS.md#round-2-prefill-and-concurrent-requests-2026-10-12).

NVFP4:

| Build | Prefill 2K | 8K | 32K | ms/step 2K | 8K | 32K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 3,660 | 3,270 | 2,530 | 122.0 | 135.8 | 178.4 |
| 128-row expert groups for prompt chunks | +7.4% | +6.4% | +4.4% | +0.1% | +0.2% | −0.3% |
| Norm/RoPE: one block per token | +5.8% | +4.6% | +3.0% | −0.2% | −0.3% | −0.4% |
| 8192-token prompt chunks | −0.6% | +5.8% | +3.8% | +0.5% | +0.6% | +0.2% |
| **All retained** (with the sessions engine) | **+11.4%** | **+18.2%** | **+12.5%** | **−0.1%** | **0.0%** | **−0.2%** |
| All retained, absolute | 4,077 | 3,866 | 2,845 | 121.9 | 135.7 | 178.1 |

FP8:

| Build | Prefill 2K | 8K | 32K | ms/step 2K | 8K | 32K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 3,331 | 3,012 | 2,358 | 138.5 | 153.2 | 196.1 |
| 128-row expert groups for prompt chunks | +10.1% | +9.0% | +6.9% | +0.1% | 0.0% | 0.0% |
| Norm/RoPE: one block per token | +5.1% | +4.0% | +3.0% | −0.3% | −0.2% | 0.0% |
| 8192-token prompt chunks | +0.5% | +6.1% | +3.5% | +0.2% | +0.9% | +1.1% |
| **All retained** (with the sessions engine) | **+15.9%** | **+21.5%** | **+14.5%** | **+0.1%** | **+1.1%** | **+1.0%** |
| All retained, absolute | 3,861 | 3,659 | 2,701 | 138.7 | 154.9 | 198.1 |

A 2K prompt is one chunk either way, so the chunk size changes nothing
there. The 128-row groups and the RoPE kernel are bit-identical to the
baseline; larger chunks change only the order of floating-point sums in
attention. Single-request denoising steps are unchanged within about 1%
(FP8 at 8K/32K, +1%, comes with the larger chunks).

## Concurrent requests

`gufo-diffusion-gemma throughput`: 8 fixed-length requests of 1,024 tokens
(end tokens suppressed), each on its own slice of the book so that none
reuses another's prompt, run by as many client threads as sessions. NVFP4.
Total output tok/s over the run; per-request ms per step is the wall time
of the batched steps it took part in.

| Prompt | Sessions | Total tok/s | Per-request tok/s | ms / step | First block s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2K | 1 | 55.2 | 55.2 | 125.5 | 5.31 |
| 2K | 2 | 60.7 (+10%) | 30.6 | 231.4 | 9.46 |
| 2K | 4 | 65.9 (+19%) | 17.2 | 416.9 | 17.37 |
| 8K | 1 | 49.3 | 49.3 | 137.0 | 7.41 |
| 8K | 2 | 51.0 (+3%) | 26.5 | 260.1 | 13.14 |
| 8K | 4 | 54.0 (+10%) | 14.7 | 479.7 | 25.80 |

Short requests gain more, because their prompts are encoded in the same
passes as the other requests' steps. 400 recorded MMLU requests (5-shot
prompts of about 400 tokens, mostly reused, single-letter answers), four
client threads against `gufo serve`, NVFP4:

| Server | Requests / s |
| --- | ---: |
| Round 1 (one request at a time) | 3.18 |
| Round 2, `--sessions 1` | 3.20 |
| Round 2, `--sessions 4`, prompts encoded between steps | 3.51 |
| Round 2, `--sessions 4`, prompts encoded within steps | **3.80 (+19%)** |

One canvas already keeps the matrix units busy: four canvases cost 3.15
single steps at 2K (only the expert products amortise; see
[EXPERIMENTS.md](EXPERIMENTS.md#concurrent-requests)), and at 8K attention
grows with each canvas's context and each new request's prompt is encoded
between steps. Sessions mainly let several users and agents share one
server with separate caches; the throughput gain is modest.

## Ablation (2026-10-11, evening)

Each change alone on the afternoon build (history below), then all
retained changes, with `gufo-diffusion-gemma bench` (story continuation, 512
tokens out, best of two interleaved rounds). FP8; prefill tok/s and ms per
denoising step, change against the baseline:

| Build | Prefill 512 | 8K | 32K | ms/step 512 | 8K | 32K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 2,334 | 2,685 | 2,169 | 150.3 | 167.9 | 209.7 |
| Dense kernel for the canvas | −0.2% | 0.0% | +0.1% | −10.0% | −9.8% | −7.3% |
| Same kernel for prompt chunks (no hipBLASLt) | +15.0% | +10.5% | +6.1% | −0.1% | +0.6% | +2.4% |
| Expert grouping, combine in the layer finish | +0.7% | +0.3% | −1.9% | −0.9% | +0.5% | +1.9% |
| Fused softmax (rejected) | −1.7% | −7.9% | −12.5% | +4.2% | +8.0% | +13.2% |
| Attention restructure (rejected) | +0.2% | −2.4% | −4.2% | +0.2% | +0.4% | +1.1% |
| **All retained** | **+16.6%** | **+13.2%** | **+9.3%** | **−11.4%** | **−9.7%** | **−5.7%** |

BF16, all retained: prefill +24.5% / +36.8% / +29.7%, steps −15.4% / −12.8% /
−12.4% (the expert spill fix alone gives +12 to +23% prefill). Q8_0 matches
FP8 within 2 points. The MLP activation in the gate/up epilogue (last
change) is bit-identical and within noise (+0.3% prefill, −0.1% per step).
Kernel-level numbers are in [EXPERIMENTS.md](EXPERIMENTS.md).

NVFP4 with the same tool (2026-10-12): prefill 3,164 / 3,311 / 2,585 tok/s
and 116 / 134 / 174 ms per step at 512 / 8K / 32K, against FP8's 133 / 152 /
198 ms. Only the expert products change: on recorded routing at 256 canvas
rows, the gate/up product takes 1.16 ms per layer against FP8's 1.61 ms.

## Where the time goes

FP8 or Q8_0, 512-token context (`gufo-diffusion-gemma bench --profile`), per
denoising step of about 134 ms: expert products about 72 ms (real routing
activates 77 of 128 experts per layer, 14.6 GB of 8-bit weights per step,
read at 180–200 GB/s, about 83% of DRAM bandwidth), dense products about
40 ms including the LM head and self-conditioning (36–47 TFLOPS, 72% of the
matrix floor), attention about 10 ms, sampler, norms, router and layer
finish about 13 ms. The hardware floor for a step is about 95 ms. At a 32K
prompt, global attention adds about 58 ms per step (about 24 TFLOPS).
Prefill at 32K ran at about 40% of its floor (13.5 s against 5.4 s) before
round 2 and at about 48% after it (NVFP4, 11.3 s). Global attention is then
34% of it (3.8 s, about 23 TFLOPS) and the expert products 27% (3.1 s,
about 30 TFLOPS); see [EXPERIMENTS.md](EXPERIMENTS.md).

## History

### Before the dense kernel (2026-10-11, afternoon)

Same protocol; hipBLASLt for prompt chunks with `GUFO_HIPBLASLT_PLAN_CACHE`
set (plans tuned by a first start, restored by the measured one).

#### Short prompt (PG19 test)

A 508-token prompt asking to continue a passage from the PG19 set, streamed
Chat Completions, `max_tokens` 512, seeds fixed; mean of 3 runs.

| Setup | Time to first token s | Tokens / s (whole request) | Tokens / s (denoising) | ms / step | Reply tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gufo Q8_0 | 4.91 | 45.6 | 47.7 | 153 | 357 |
| Gufo FP8 | 5.04 | 54.7 | 56.7 | 150 | 512 |
| Gufo BF16 | 6.77 | 39.9 | 41.6 | 216 | 512 |
| llama.cpp Q8_0 | 12.34 | 20.7 | 20.7 | 386 | 512 |

Tokens per second depends on the engine's time per step and on how many
tokens each step completes; the second depends on the text and the weights
(Q8_0 stopped after 357 tokens here). Compare engines by ms / step.

#### Long prompts

The first *N* tokens of a book in one user turn, 512 tokens out, one
request per length (two for 2K–32K, which agree within 1%).

| Prompt tokens | Prefill tok/s Q8_0 | FP8 | BF16 | llama.cpp |
| ---: | ---: | ---: | ---: | ---: |
| 2,261 | 2,696 | 2,648 | 2,149 | 978 |
| 8,500 | 2,613 | 2,568 | 2,171 | 713 |
| 33,105 | 2,169 | 2,136 | 1,875 | 445 |
| 65,389 | 1,761 | 1,694 | 1,540 | 310 |
| 121,294 | 1,310 | 1,255 | 1,168 | 201 |

| Prompt tokens | Time to first token s Q8_0 | FP8 | BF16 | llama.cpp |
| ---: | ---: | ---: | ---: | ---: |
| 2,261 | 6.5 | 6.2 | 7.6 | 18.8 |
| 8,500 | 10.6 | 9.9 | 12.9 | 27.2 |
| 33,105 | 23.5 | 21.9 | 25.6 | 67.9 |
| 65,389 | 48.6 | 50.3 | 56.4 | 142.7 |
| 121,294 | 110.2 | 114.2 | 121.3 | 368.3 |

The first visible tokens arrive with the first 256-token canvas: prefill
plus one block of denoising.

| Prompt tokens | ms / step Q8_0 | FP8 | BF16 | llama.cpp |
| ---: | ---: | ---: | ---: | ---: |
| 2,261 | 157 | 156 | 217 | 393 |
| 8,500 | 166 | 168 | 223 | 441 |
| 33,105 | 209 | 213 | 271 | 646 |
| 65,389 | 269 | 276 | 331 | 937 |
| 121,294 | 362 | 379 | 430 | 1,365 |

Startup to the first accepted request: 8–12 s with a plan database (load
5–9 s, warmup 2.4–3.0 s), 16–20 s on the first start that fills it (warmup
about 10 s of tuning).


### vLLM (2026-10-01) and the first Gufo build (2026-10-10)

The same checkpoint under vLLM 0.29 (FP16, eager, Triton attention/MoE, one
sequence), measured 2026-10-01 by `diffusion-fp16-vllm/scripts/context_benchmark.py`
with a synthetic filler prompt and 512-token replies (medians, end-to-end over
HTTP including prefill):

| Input tokens | vLLM first visible s | vLLM total s | vLLM tok/s (total) |
| ---: | ---: | ---: | ---: |
| 1,528 | 11.04 | 18.13 | 28.2 |
| 3,556 | 14.31 | 26.60 | 19.3 |
| 7,677 | 29.71 | 44.99 | 11.4 |
| 15,853 | 60.96 | 85.18 | 6.0 |
| 24,059 | 109.80 | 136.58 | 3.8 |
| 32,239 | 199.51 | 250.82 | 2.0 |

The first Gufo build's end-to-end HTTP results at the same sizes, BF16:

`gufo serve llm --context 40960`, streamed Chat Completions, `max_tokens`
512, seeds 100–102, the Pride and Prejudice continuation prompt sized by
characters (so inputs are slightly larger than the vLLM rows above). Time
from request to the first visible content and to the last token, including
prefill; medians of 3. Every request returned 512 tokens. Script:
`http_bench.py` (kept with the run artifacts, not in the repository).

| Input tokens | First visible s | Total s | tok/s (512 / total) | vLLM total s at similar input |
| ---: | ---: | ---: | ---: | ---: |
| 1,696 | 12.67 | 20.61 | 24.8 | 18.13 (1,528) |
| 3,762 | 13.52 | 23.14 | 22.1 | 26.60 (3,556) |
| 7,977 | 15.51 | 24.44 | 21.0 | 44.99 (7,677) |
| 16,229 | 20.78 | 33.53 | 15.3 | 85.18 (15,853) |
| 24,471 | 24.57 | 37.39 | 13.7 | 136.58 (24,059) |
| 32,576 | 29.87 | 40.57 | 12.6 | 250.82 (32,239) |

The prompts differ (book text here, synthetic filler for vLLM), and a
diffusion model's step count depends on the text, so short-context rows are
comparable only roughly; the long-context gap is prefill (vLLM about 160
tok/s at 32K, Gufo about 1,600).

Startup: weights load in about 5 s from a warm page cache (14 s cold),
then 54 s of GEMM tuning before the server listens.

### Plan cache and self-conditioning kernel (2026-10-10)

Measured when both were server options (`--diffusion-gemm-cache`,
`--diffusion-soft-embed-kernel`); the kernel is now always used and the
plans go to Gufo's shared plan database.

Both options alone and together against neither (Strix Halo, gfx1151,
ROCm 7.2.3, `--context 131072`). Startup is container start to first
accepted request; cache variants were measured on a second start, with the
plan file written by the first (60–62 s).

| Weights | Variant | Ready s | Warmup s | ms / step (508-token prompt) |
| --- | --- | ---: | ---: | ---: |
| Q8_0 | neither | 66 | 53.7 | 197.8 |
| Q8_0 | `--diffusion-gemm-cache` | 6 | 2.6 | 193.1 |
| Q8_0 | `--diffusion-soft-embed-kernel` | 14 | 10.2 | 180.3 |
| Q8_0 | both | 6 | 2.6 | 181.3 |
| BF16 | neither | 66 | 53.9 | 261.9 |
| BF16 | `--diffusion-gemm-cache` | 10 | 3.3 | 266.3 |
| BF16 | `--diffusion-soft-embed-kernel` | 18 | 11.0 | 247.5 |
| BF16 | both | 10 | 3.2 | 243.7 |

The kernel saves 12–22 ms per denoising step (16 ms on average), and
6–9% at short prompts. Across the 2K–121K prompt sweep it stays 15–35 ms
faster on Q8_0 and 4–21 ms faster on BF16. The plan cache does not change speed: it fixes
whichever plans the writing start tuned, and its ±2% is tuning variance.
Tokens per second on the PG19 protocol is a poor comparison here, since the
kernel's different summation order changes the step count and where the
text stops (BF16 runs ended at 394–512 tokens).

Accuracy of the product against float64: max error 2.74e-5 for both
hipBLASLt and the kernel, mean 3.54e-6 and 3.45e-6. With plans pinned,
the kernel moves step-2 logits by mean KL 0.0105 from the hipBLASLt path;
two cache-less starts differ from each other by 0.041.
