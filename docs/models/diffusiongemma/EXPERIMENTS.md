# DiffusionGemma experiments

Short record of retained and rejected decisions (2026-10-10 and 11, gfx1151,
ROCm 7.2.3). Device read bandwidth measured at 242 GB/s.

## Expert GEMMs (`moe.hip`)

Synthetic routing, 256 canvas rows over 75 of 128 experts (the measured
average for canvas passes is 67–75), and a 2048-row prefill chunk over all
experts:

| Variant | gate/up 256 rows | down 256 rows | gate/up 2048 rows | down 2048 rows |
| --- | ---: | ---: | ---: | ---: |
| LDS-staged weights, `__syncthreads` (first version) | 3.09 ms, 193 GB/s | 1.46 ms, 204 GB/s | 13.5 ms, 9.6 TF | 4.62 ms, 14.1 TF |
| Weights straight to WMMA fragments, LDS for A | 10.9 ms | 1.47 ms | 51.7 ms | 4.86 ms |
| Both operands straight to fragments, no barriers | 28.4 ms | 1.93 ms | 74.3 ms | 8.34 ms |
| **LDS-staged, double-buffered, LDS-only barrier** | 2.87 ms, 207 GB/s | 1.43 ms, 208 GB/s | 10.9 ms, 12.0 TF | 4.86 ms, 13.4 TF |
| **+ 2x2 WMMA tile per wave (retained)** | 2.89 ms, 206 GB/s | 1.41 ms, 211 GB/s | 10.4 ms, 12.5 TF | 4.06 ms, 16.0 TF |

- Rejected: loading weight fragments directly. A fragment is 32 bytes from
  each of 16 rows 5.6 KiB apart, so each 128-byte line is consumed over four
  separate loads and is evicted in between under load.
- Retained: `s_waitcnt lgkmcnt(0); s_barrier` instead of `__syncthreads()`.
  A workgroup barrier otherwise waits for every outstanding global load,
  which drains the next stage's register prefetch at each stage.

## Dense GEMMs (`gemm.cpp`, removed 2026-10-11)

hipBLASLt products were replaced by `kernels::Linear` (below); kept for the
record:

- hipBLASLt's first heuristic candidate is often slow on gfx1151; the
  wrapper times up to 64 candidates per shape (and split-K variants for
  narrow outputs) once, at warmup.
- Rejected: BF16-output hipBLASLt kernels. Faster (o_proj 0.40 → 0.25 ms,
  LM head 16.3 → 14.4 ms at 256 rows) but they do not equal FP32
  accumulation with round-to-nearest-even: feeding the same values through
  an FP32-output GEMM plus an exact conversion restored the reference
  agreement (self-conditioned KL 0.031 → 0.0118).
- Split-K (FP32 partials, FP32 reduction) helps the self-conditioning
  product (K = 262,144; 22.1 → 21.2 ms) and o_proj (0.38 → 0.32 ms). The
  global o_proj (K = 8192) found no faster split.

## Attention (`attention.hip`)

- Split-key decoding: the canvas pass has only 4 query tiles x 16 heads =
  64 blocks. Splitting the keys (up to 16 slices, merged by a second kernel)
  cut 32K-context denoising steps from 389 to 360 ms.
- Register-resident query fragments with K/V staged once per block (head
  dimension sliced across waves) replaced per-tile query reloads; measured
  effect was small (32K prefill 21.3 → 21.1 s). Global-layer attention runs
  at about 14–15 TFLOPS and is the main cost of long contexts.
- Retained (2026-10-11): register prefetch of the next K/V tile, issued
  after the staging barrier. Global attention, 16 heads over 2 K/V heads,
  head dimension 512 (standalone kernel benchmark): a 32K canvas
  step 19.0 → 13.7 ms per layer, a 2048-query prefill chunk at 32K
  173 → 105 ms. Output checksums unchanged. The kernel read only about
  7 GB/s of unique K/V before: it was waiting on load latency, not bandwidth.
- Retained: no barrier between the softmax's row-max read and its update
  (the threads of a row are one wave); 1–2%, bit-identical.
- Retained: `amdgpu_waves_per_eu(8)` for head dimension 512, so two 16-wave
  blocks share a WGP (192 registers, 52 bytes/lane of spill): 13.7 → 11.7 ms
  and 105 → 86 ms. Model level, Q8_0 at 32K: 250 → 209 ms/step, prefill
  18.0 → 14.8 s.
- Rejected: reading the softmax correction from LDS inside the P·V loop to
  remove the spill (no change). Deferred: V stored transposed in the cache
  (about 6% here, but a cache layout change).
- Hardware-counter collection (rocprofv3 `--pmc`) on a 32K model run
  serialises every launch and did not finish in 9 hours; time the attention
  kernel on its own instead.

## Canvas products (`dense.hip`) and plan tuning (`gemm.cpp`, before 2026-10-11 evening)

- Retained: the hipBLASLt tuner clears its 64 MB workspace before each
  timed run. Candidates timed with cache-hot weights were up to 3x slower in
  the model (o_proj 0.29 ms tuned, 0.97 ms in the model); Q8_0 ms/step at
  512 tokens 184 → 173.
- Retained: one kernel for every 256-row canvas product, each workgroup
  covering all rows (weights read once) with both operands blocked
  (`[k/32][256][32]`, `[n/128][k/32][128][32]`). Row-major operands ran at
  about half speed (channel camping, 64 bytes per row per stage). 173 → 153
  ms/step at 512 tokens; 4.8 GB of blocked weight copies.
- Rejected: 8-bit dense weights for the canvas products. At 256 rows the
  products sit at the ridge point (QKV 46 MB in 0.19 ms at 241 GB/s against
  0.21 ms of matrix math), so halving the bytes saves little.
- Rejected: prefill chunks of 4096 and 8192 rows (8K/32K prefill +3%/+5%)
  for several hundred MB more scratch.
- Retained: the self-conditioning kernel (`soft_embed.hip`) as the only
  path (was optional): 6–9% per step at short prompts, and it removes the
  262,144-deep hipBLASLt shape from startup tuning.
- Retained: tuned plans go to Gufo's shared hipBLASLt plan database
  (`GUFO_HIPBLASLT_PLAN_CACHE`, schema 2 adds the split-K count) instead of
  a model-specific file and option.

## Expert GEMMs, prefill (2026-10-11)

- Rejected: a 128-row tile (4 x 2 waves of 32 x 64, depth 32) for prompt
  chunks. 2048-row chunk over 128 experts: FP8 gate/up 6.45 → 5.89 ms and
  down 3.30 → 2.80 ms, Q8_0 gate/up 6.12 → 6.42 ms (slower; the per-block
  scaling dominates), BF16 gate/up 10.5 → 14.1 ms.

## One dense kernel for every product (2026-10-11, evening)

`kernels::Linear` (`dense.hip`) now runs every dense product: canvas,
prompt chunks, router, self-conditioning MLP and LM head. Row-major
activations, weights blocked in place at load as `[n/128][k/64][128][64]`,
8 waves (4 x 2) over a 256 x 128 tile (128 x 128 for the 2816-wide
outputs), 64-deep stages through one LDS buffer with register prefetch.
Standalone kernel benchmark on the model's shapes, weights cycled from memory,
best of 5 batches after warm-up:

| Product | Canvas before | Canvas now | 2048 rows, hipBLASLt | 2048 rows now |
| --- | ---: | ---: | ---: | ---: |
| QKV sliding (8192 x 2816) | 0.419 ms | 0.325 ms | 3.03 ms | 2.08 ms |
| QKV global (9216 x 2816) | 0.408 ms | 0.327 ms | 3.35 ms | 2.37 ms |
| O sliding (2816 x 4096) | 0.263 ms | 0.159 ms | 2.10 ms | 1.19 ms |
| O global (2816 x 8192) | 0.482 ms | 0.429 ms | 4.59 ms | 2.68 ms |
| MLP gate/up (4224 x 2816) | 0.205 ms | 0.148 ms | 1.55 ms | 1.04 ms |
| MLP down (2816 x 2112) | 0.129 ms | 0.085 ms | 0.81 ms | 0.54 ms |
| LM head (262144 x 2816) | 10.6 ms | 8.06 ms (46.9 TFLOPS) | | |

- Retained: native vector types (`ext_vector_type(4)`) for the prefetch
  registers. HIP's `uint4` kept them in scratch, and the compiler waited for
  each prefetch at once to spill it; about 15% on every shape. The same fix
  sped up the self-conditioning product 10.3 → 9.15 ms (bit-identical) and
  the BF16 expert kernels (below).
- Retained: unconditional prefetch (rows past the end repeat the last row,
  whose products are not stored; the last stage reloads itself). A
  conditional zero also forced scratch.
- Retained: `__builtin_amdgcn_sched_barrier` between the four 16-deep
  sub-steps of a stage: the compiler otherwise hoists all fragment loads and
  spills; 10–25% per shape.
- Retained: 64-deep stages with one LDS buffer (two barriers per stage)
  over 32-deep double buffering; row-major activations, so no blocking pass.
- Rejected: stream-K scheduling (40 persistent workgroups with ordered
  fix-up) to fill the second wave of 64-tile products: 0.50 ms against 0.35
  ms for the plain grid. Split-K for the 2816-wide products also lost to
  128-row tiles.
- Removing hipBLASLt drops startup tuning (about 54 s cold, 3–12 s with a
  plan database) and the row-major copies of the dense weights (3.3 GB):
  the matrices are blocked in place at load; only the LM head keeps a
  blocked copy of the tied embedding.

## Experts and the layer finish (2026-10-11, evening)

Real canvas routing (captured from a 512-token story run and replayed through
the expert kernels on their own, 256 canvas rows, Q8_0): 77 active
experts per layer on average (not the 66 of earlier estimates), 60% of
them with fewer than 16 rows and one with about 190. Expert weight traffic
is then about 14.6 GB per Q8_0 step; the kernels read it at 180–200 GB/s.

- Retained: Dispatch lists each expert's 64-row groups and the expert
  kernels take one group per workgroup, instead of one expert per workgroup
  looping over its groups. Neutral for Q8_0/FP8 on real routing, slightly
  slower (2–6%) on uniform 2048-row chunks; kept for the simpler kernel.
- Retained: expert down-projection output in BF16, the reference's
  checkpoint dtype for the expert projection, and the combine folded into
  FinishLayer (no Combine kernel or `moe_out`): combine + finish 5.1 → 3.6
  ms per step.
- Retained: the spill fix above for BF16 experts: gate/up 2.58 → 2.43 ms
  (256 rows), 10.4 → 5.86 ms (2048 rows).
- Rejected: prefetching two stages ahead (more loads in flight): no change
  on real routing.

## Denoiser head and sampler (2026-10-11, evening)

- Rejected: the sampler fused into the LM head's epilogue (softcap,
  BF16 logits, per-tile max/argmax/Gumbel/softmax sums merged per row, then
  an in-place probabilities pass), which avoids writing and reading 268 MB of
  FP32 logits per step. Head + sampler 15.5 ms against 8.2 + 5.5 ms
  separately: the sampler is bound by its per-logit transcendental math
  (tanh, two logs, two exps), which the fusion moves onto the matrix
  kernel's critical path; a first version without LDS staging spilled 544
  bytes per lane and took 18.3 ms.

## Attention (2026-10-11, evening)

Phase timing with in-kernel cycle counters (global decode, 32K keys):
staging 14%, QK and its barrier 36%, softmax 22%, P·V 11%, the loop's
barrier 17%. Standalone kernel benchmark, interleaved, best of 3;
end-to-end results are in [BENCHMARKS.md](BENCHMARKS.md#ablation-2026-10-11-evening).

| Variant | Global decode 32K | Global prefill chunk 32K | Sliding decode | Sliding prefill chunk |
| --- | ---: | ---: | ---: | ---: |
| Retained kernel | 11.74 ms | 85.4 ms | 0.306 ms | 1.80 ms |
| Fused softmax: S^T = K Q^T, softmax in registers, P straight into O^T = V^T P^T, one barrier fewer | 14.96 ms | 112.7 ms | 0.584 ms | 3.28 ms |
| Native/unconditional prefetch, DPP row reductions, paired V stores | 11.40 ms | 86.8 ms | 0.306 ms | 1.76 ms |
| + two query groups per wave over 64 dimensions (half the K/V fragment reads), K and V sharing one LDS region | 11.65 ms | 89.3 ms | 0.299 ms | 1.73 ms |

- Rejected: the fused softmax. With the head dimension split across waves,
  every wave of a query group needs the whole probability tile, so the
  softmax's exponentials are computed 4 times (head dimension 512) or twice
  (256); a fast `exp2` did not change that (14.6 ms).
- Rejected: the smaller fixes and the two-group layout: within ±3% (noise
  level) at kernel level. Neither LDS bandwidth (halved fragment reads) nor
  the softmax's arithmetic is the limit; the barrier-separated phases are,
  with at most two 63 KB blocks per WGP to overlap them. Larger key tiles do
  not fit in 64 KB of LDS at head dimension 512.
- Rejected: more key splits for short decodes (256/128/64-key minimum):
  no gain or slower.

## Round 2: prefill and concurrent requests (2026-10-12)

The engine profile now splits each attention block into the QKV product,
norm/RoPE, the attention kernel and the output product. NVFP4, 32K prompt,
before this round (13.0 s of prefill): global attention 3.71 s, expert
products 4.06 s, the sliding layers' QKV product 0.86 s and their norm/RoPE
0.63 s, sliding attention 0.72 s, output products 0.97 s, dense MLP 0.90 s.

- Retained: 128-row expert groups for prompt chunks (`moe.hip`). A 2048-row
  chunk averages 128 rows per expert, so 64-row groups widened each weight
  tile twice. The 128-row kernel keeps the 64-row kernel's per-wave tile
  (32 x 32, 16 waves, 512 threads) with one LDS buffer; an 8-wave version
  with 32 x 64 per wave spilled 70-122 registers and was 30% slower.
  Prefill +4 to +9% (NVFP4, FP8, BF16); Q8_0 1-3% slower, so it keeps 64.
  Bit-identical.
- Retained: one block per token for the Q/K/V norm and RoPE
  (`QkvNormRopeKernel`). The per-head kernel launched 65,536 128-thread
  blocks per 2048-row chunk, each recomputing `sincosf` of large angles
  (the slow argument-reduction path) for the same position. The angles
  are now computed once per token into LDS and each wave takes one head;
  the squares are summed in the old kernel's order, so the outputs are
  bit-identical. Sliding norm/RoPE 0.63 s -> 0.21 s at 32K.
- Retained: 8192-token prompt chunks (was 2048). With 128-row groups,
  larger chunks fill the expert groups: 8K prefill +6.7%, 32K +4.8% (4096:
  +4.2%, +3.2%). Costs about 2.4 GB more scratch and sliding cache.
- Rejected: 256-row expert groups for 8192-row chunks (1024 threads): NVFP4
  1-5% slower, FP8 unchanged.
- Rejected, NVFP4 expert products on recorded canvas routing (gate/up
  1.16 ms, 145 GB/s; FP8 1.60 ms, 188 GB/s): prefetching two stages ahead
  (1.22 ms; the run-time ring index went to scratch) and 128-deep stages
  for gate/up (1.44 ms). With the FP4 decode removed altogether (wrong
  values) the kernel took 1.03 ms: the limit is the per-stage work
  (barriers, activation loads, matrix work over mostly empty row tiles),
  not the decode or the weight bytes.
- Not attempted: merging the sampler's passes. `SampleRows` reads each
  row's 1 MB of FP32 logits three times (max and Gumbel, sums, the
  self-conditioning probabilities): about 940 MB per canvas with the
  probability writes, about 4 of its 6 ms. An online softmax would drop one
  pass (about 1 ms per step, 1%) but change the order of the sums, so the
  sampled canvases would no longer match the current ones bit for bit.
- Not attempted: the global attention kernel (34% of 32K prefill after
  this round). Its tiles are limited by LDS at head dimension 512 (the
  key tile, transposed value tile and the cross-wave score partials fill
  64 KB), and packing query heads that share a KV head does not change
  the K/V traffic or staging per block. Transposed V in the cache remains
  the deferred 6% option.

### Concurrent requests

The engine holds `--sessions` caches. Each session's sliding caches hold
the window, the reuse history and one canvas; prompt chunks run in shared
stage caches, copied in before and out after (with one session the
sliding caches hold a whole chunk, as before). A worker thread admits
requests to free sessions (longest reusable prefix first), runs one prompt
chunk (2048 tokens while others denoise) and then one denoising step over
every active canvas: one forward pass over the stacked rows, with
attention, norm/RoPE writes and sampling per canvas. With one session the
outputs are bit-identical to the single-session engine.

Per-step cost, NVFP4, 2K context, the throughput tool (`--profile`):

| Phase | 1 canvas | 4 canvases | Ratio |
| --- | ---: | ---: | ---: |
| Expert products | 54.7 ms | 134.5 ms | 2.46 |
| Self-conditioning, LM head, sampler | 24.2 ms | 89.8 ms | 3.71 |
| Dense products and MLP | 25.3 ms | 92.1 ms | 3.64 |
| Attention (incl. norm/RoPE) | 16.3 ms | 64.1 ms | 3.93 |
| Layer finish, post-attention, router | 7.0 ms | 21.2 ms | 3.03 |
| Whole step | 127.7 ms | 402.2 ms | 3.15 |

One 256-row canvas already keeps the matrix units busy (the dense products
run at 36-47 TFLOPS), so only the expert products amortise. Four canvases
cost 3.15 steps: total throughput can rise by at most about 1.27x.

- Retained: 128-row expert groups once experts average 32 rows (two or more
  canvases): step time -1.7% (two canvases), -3.4% (four).
- Retained: prompt chunks and committed canvases in the batched pass. Each
  pass runs every denoising canvas and then, as causal segments, the
  finished canvases to commit and prompt chunks within a row budget (2048
  rows while others denoise, at most one chunk through the stage caches);
  a finished canvas is committed in the next pass, which keeps each
  request's random draws in order. Self-conditioning, the LM head and the
  sampler run over the canvas rows only. With one session nothing is
  mixed and the outputs are bit-identical. 400 recorded MMLU requests
  (5-shot prompts, single-letter answers) from four clients: 3.51 -> 3.80
  requests/s (the round-1 server: 3.18; one session: 3.20). Fixed-length
  1,024-token requests: within ±2% per canvas step (mixing changes the order
  of some sums, so the trajectories and step counts differ).
