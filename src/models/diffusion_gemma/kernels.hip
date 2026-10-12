#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cfloat>
#include <stdexcept>
#include <string>

#include "src/models/diffusion_gemma/device.hpp"
#include "src/models/diffusion_gemma/hip_check.hpp"
#include "src/models/diffusion_gemma/kernels.hpp"

namespace gufo::models::diffusion_gemma::kernels {
namespace {

constexpr int kThreads = 256;

// Sum across a 256-thread block. `shared` holds at least 8 floats.
__device__ float BlockSum(float value, float* shared) {
  value = WaveSum(value);
  const int lane = threadIdx.x % 32, wave = threadIdx.x / 32;
  __syncthreads();
  if (lane == 0)
    shared[wave] = value;
  __syncthreads();
  float total = 0;
  for (int w = 0; w < static_cast<int>(blockDim.x) / 32; ++w)
    total += shared[w];
  return total;
}

__device__ float RmsScale(float sum_squares, int n, float eps) {
  return 1.0F / sqrtf(sum_squares / static_cast<float>(n) + eps);
}

__global__ void __launch_bounds__(kThreads)
    EmbedKernel(const std::int32_t* ids, const bf16_t* embed, float scale,
                float* out, int hidden) {
  const int t = blockIdx.x;
  const bf16_t* row = embed + static_cast<std::size_t>(ids[t]) * hidden;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x)
    out[static_cast<std::size_t>(t) * hidden + i] =
        RoundBf16(ToFloat(row[i]) * scale);
}

__global__ void __launch_bounds__(kThreads)
    RmsNormKernel(const float* x, const bf16_t* w, bf16_t* out, int hidden,
                  float eps) {
  __shared__ float shared[8];
  const float* row = x + static_cast<std::size_t>(blockIdx.x) * hidden;
  float sum = 0;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x)
    sum += row[i] * row[i];
  const float r = RmsScale(BlockSum(sum, shared), hidden, eps);
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    const float scale = w ? ToFloat(w[i]) : 1.0F;
    out[static_cast<std::size_t>(blockIdx.x) * hidden + i] =
        ToBf16(row[i] * r * scale);
  }
}

constexpr int kMaxPerThread = 12;  // hidden 2816 = 11 * 256.

__global__ void __launch_bounds__(kThreads)
    PostAttentionKernel(float* h, const float* o, const bf16_t* post_attention,
                        const bf16_t* pre_ff, const bf16_t* pre_ff_2,
                        const bf16_t* router_scale, bf16_t* ff, bf16_t* ff2,
                        bf16_t* rout, int hidden, float eps) {
  __shared__ float shared[8];
  const std::size_t base = static_cast<std::size_t>(blockIdx.x) * hidden;
  float ov[kMaxPerThread], hv[kMaxPerThread];
  float sum = 0;
#pragma unroll
  for (int j = 0; j < kMaxPerThread; ++j) {
    const int i = threadIdx.x + j * kThreads;
    ov[j] = i < hidden ? o[base + i] : 0.0F;
    sum += ov[j] * ov[j];
  }
  const float ro = RmsScale(BlockSum(sum, shared), hidden, eps);
  sum = 0;
#pragma unroll
  for (int j = 0; j < kMaxPerThread; ++j) {
    const int i = threadIdx.x + j * kThreads;
    if (i < hidden) {
      // Checkpoint-dtype boundaries: the normed branch and the sum are BF16.
      hv[j] = RoundBf16(h[base + i] +
                        RoundBf16(ov[j] * ro * ToFloat(post_attention[i])));
      h[base + i] = hv[j];
    } else {
      hv[j] = 0;
    }
    sum += hv[j] * hv[j];
  }
  const float rh = RmsScale(BlockSum(sum, shared), hidden, eps);
  const float root = rsqrtf(static_cast<float>(hidden));
#pragma unroll
  for (int j = 0; j < kMaxPerThread; ++j) {
    const int i = threadIdx.x + j * kThreads;
    if (i < hidden) {
      const float n = hv[j] * rh;
      ff[base + i] = ToBf16(n * ToFloat(pre_ff[i]));
      ff2[base + i] = ToBf16(n * ToFloat(pre_ff_2[i]));
      rout[base + i] =
          ToBf16(RoundBf16(RoundBf16(n) * ToFloat(router_scale[i])) * root);
    }
  }
}

__global__ void __launch_bounds__(kThreads)
    FinishLayerKernel(float* h, const float* mlp, ExpertOutputs moe,
                      const bf16_t* post_ff_1, const bf16_t* post_ff_2,
                      const bf16_t* post_ff, float layer_scalar, int hidden,
                      float eps) {
  __shared__ float shared[8];
  const int t = blockIdx.x;
  const std::size_t base = static_cast<std::size_t>(t) * hidden;
  float a[kMaxPerThread], b[kMaxPerThread];
  float sa = 0, sb = 0;
#pragma unroll
  for (int j = 0; j < kMaxPerThread; ++j) {
    const int i = threadIdx.x + j * kThreads;
    a[j] = i < hidden ? mlp[base + i] : 0.0F;
    // The row's expert outputs, combined in assignment order.
    b[j] = 0;
    if (i < hidden)
      for (int e = 0; e < moe.k; ++e)
        b[j] += moe.weight[t * moe.k + e] *
                ToFloat(moe.expert_out[static_cast<std::size_t>(
                                           moe.slot[t * moe.k + e]) *
                                           hidden +
                                       i]);
    sa += a[j] * a[j];
    sb += b[j] * b[j];
  }
  const float ra = RmsScale(BlockSum(sa, shared), hidden, eps);
  const float rb = RmsScale(BlockSum(sb, shared), hidden, eps);
  float sc = 0;
#pragma unroll
  for (int j = 0; j < kMaxPerThread; ++j) {
    const int i = threadIdx.x + j * kThreads;
    if (i < hidden)
      a[j] = RoundBf16(RoundBf16(a[j] * ra * ToFloat(post_ff_1[i])) +
                       RoundBf16(b[j] * rb * ToFloat(post_ff_2[i])));
    else
      a[j] = 0;
    sc += a[j] * a[j];
  }
  const float rc = RmsScale(BlockSum(sc, shared), hidden, eps);
#pragma unroll
  for (int j = 0; j < kMaxPerThread; ++j) {
    const int i = threadIdx.x + j * kThreads;
    if (i < hidden)
      h[base + i] = RoundBf16(
          RoundBf16(h[base + i] + RoundBf16(a[j] * rc * ToFloat(post_ff[i]))) *
          layer_scalar);
  }
}

// One block per (token, head slot); blockDim = head_dim / 2, each thread
// owns the RoPE pair (i, i + head_dim/2).
__global__ void QkvNormRopeKernel(const float* qkv, const bf16_t* q_norm,
                                  const bf16_t* k_norm, const float* inv_freq,
                                  bf16_t* q_out, bf16_t* k_cache,
                                  bf16_t* v_cache, int q_heads, int kv_heads,
                                  int hd, bool shared_kv, int first_position,
                                  int first_slot, int capacity, float eps) {
  __shared__ float shared[8];
  const int t = blockIdx.x, slot = blockIdx.y, i = threadIdx.x, half = hd / 2;
  const int row_width = (q_heads + (shared_kv ? 1 : 2) * kv_heads) * hd;
  const float* row = qkv + static_cast<std::size_t>(t) * row_width;
  enum { kQ, kK, kV } kind;
  int head;
  const float* src;
  if (slot < q_heads) {
    kind = kQ;
    head = slot;
    src = row + head * hd;
  } else if (slot < q_heads + kv_heads) {
    kind = kK;
    head = slot - q_heads;
    src = row + (q_heads + head) * hd;
  } else {
    kind = kV;
    head = slot - q_heads - kv_heads;
    src = shared_kv ? row + (q_heads + head) * hd
                    : row + (q_heads + kv_heads + head) * hd;
  }
  // The projection output is a checkpoint-dtype tensor. hipBLASLt's BF16
  // output kernels do not round like this, so the product stays FP32 until
  // here.
  const float x0 = RoundBf16(src[i]), x1 = RoundBf16(src[i + half]);
  const float r = RmsScale(BlockSum(x0 * x0 + x1 * x1, shared), hd, eps);
  const int position = first_position + t;
  if (kind == kV) {
    bf16_t* dst =
        v_cache +
        (static_cast<std::size_t>(head) * capacity + first_slot + t) * hd;
    dst[i] = ToBf16(x0 * r);
    dst[i + half] = ToBf16(x1 * r);
    return;
  }
  const bf16_t* w = kind == kQ ? q_norm : k_norm;
  const float n0 = RoundBf16(x0 * r * ToFloat(w[i]));
  const float n1 = RoundBf16(x1 * r * ToFloat(w[i + half]));
  const float angle = static_cast<float>(position) * inv_freq[i];
  float s, c;
  sincosf(angle, &s, &c);
  const bf16_t o0 = ToBf16(n0 * c - n1 * s);
  const bf16_t o1 = ToBf16(n1 * c + n0 * s);
  bf16_t* dst =
      kind == kQ ? q_out + (static_cast<std::size_t>(t) * q_heads + head) * hd
                 : k_cache + (static_cast<std::size_t>(head) * capacity +
                              first_slot + t) *
                                 hd;
  dst[i] = o0;
  dst[i + half] = o1;
}

// One wave per row; experts <= 128 (4 per lane).
__global__ void RouteKernel(const float* logits, const bf16_t* per_expert,
                            std::int32_t* index, float* weight, int rows,
                            int experts, int k) {
  const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  if (row >= rows)
    return;
  float v[4];
  float m = -FLT_MAX;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int e = lane + 32 * j;
    v[j] = e < experts ? logits[static_cast<std::size_t>(row) * experts + e]
                       : -FLT_MAX;
    m = fmaxf(m, v[j]);
  }
  m = WaveMax(m);
  float sum = 0;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int e = lane + 32 * j;
    v[j] = e < experts ? expf(v[j] - m) : -1.0F;
    sum += e < experts ? v[j] : 0.0F;
  }
  sum = WaveSum(sum);
#pragma unroll
  for (int j = 0; j < 4; ++j)
    v[j] = v[j] >= 0 ? v[j] / sum : -1.0F;
  float picked_w[8];
  int picked_e[8];
  float total = 0;
  for (int s = 0; s < k; ++s) {
    // Largest probability; lowest expert index on ties.
    float best = -2.0F;
    int best_e = 1 << 30;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int e = lane + 32 * j;
      if (v[j] > best || (v[j] == best && e < best_e)) {
        best = v[j];
        best_e = e;
      }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
      const float ob = __shfl_xor(best, offset, 32);
      const int oe = __shfl_xor(best_e, offset, 32);
      if (ob > best || (ob == best && oe < best_e)) {
        best = ob;
        best_e = oe;
      }
    }
    picked_w[s] = best;
    picked_e[s] = best_e;
    total += best;
#pragma unroll
    for (int j = 0; j < 4; ++j)
      if (lane + 32 * j == best_e)
        v[j] = -2.0F;
  }
  if (lane < k) {
    float w = 0;
    int e = 0;
    for (int s = 0; s < k; ++s)
      if (s == lane) {
        w = picked_w[s];
        e = picked_e[s];
      }
    index[static_cast<std::size_t>(row) * k + lane] = e;
    weight[static_cast<std::size_t>(row) * k + lane] =
        w / total * ToFloat(per_expert[e]);
  }
}

__global__ void __launch_bounds__(1024)
    DispatchKernel(const std::int32_t* index, std::int32_t* counts,
                   std::int32_t* offsets, std::int32_t* source,
                   std::int32_t* slot, std::int32_t* groups,
                   std::int32_t* group_count, int assignments, int experts,
                   int k) {
  __shared__ int count[256], start[256], cursor[256];
  for (int e = threadIdx.x; e < experts; e += blockDim.x) {
    count[e] = 0;
    cursor[e] = 0;
  }
  __syncthreads();
  for (int a = threadIdx.x; a < assignments; a += blockDim.x)
    atomicAdd(&count[index[a]], 1);
  __syncthreads();
  if (threadIdx.x == 0) {
    // Expert offsets, and the expert products' work list: one entry per
    // 64-row group of each expert.
    int running = 0, listed = 0;
    for (int e = 0; e < experts; ++e) {
      start[e] = running;
      running += count[e];
      for (int first = 0; first < count[e]; first += kExpertGroupRows) {
        groups[2 * listed] = e;
        groups[2 * listed + 1] = first;
        ++listed;
      }
    }
    *group_count = listed;
  }
  __syncthreads();
  for (int e = threadIdx.x; e < experts; e += blockDim.x) {
    counts[e] = count[e];
    offsets[e] = start[e];
  }
  for (int a = threadIdx.x; a < assignments; a += blockDim.x) {
    const int e = index[a];
    const int position = start[e] + atomicAdd(&cursor[e], 1);
    slot[a] = position;
    source[position] = a / k;
  }
}

__device__ std::uint32_t Hash(std::uint64_t seed, std::uint32_t row,
                              std::uint32_t col) {
  std::uint64_t x = seed ^ (static_cast<std::uint64_t>(row) << 32 | col);
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ULL;
  x ^= x >> 33;
  return static_cast<std::uint32_t>(x);
}

constexpr int kRowThreads = 1024;

struct ArgBest {
  float value;
  int index;
};

__device__ ArgBest Better(ArgBest a, ArgBest b) {
  if (b.value > a.value || (b.value == a.value && b.index < a.index))
    return b;
  return a;
}

__device__ ArgBest BlockArgMax(ArgBest v, float* sv, int* si) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    ArgBest o{__shfl_xor(v.value, offset, 32), __shfl_xor(v.index, offset, 32)};
    v = Better(v, o);
  }
  const int lane = threadIdx.x % 32, wave = threadIdx.x / 32;
  __syncthreads();
  if (lane == 0) {
    sv[wave] = v.value;
    si[wave] = v.index;
  }
  __syncthreads();
  ArgBest best{sv[0], si[0]};
  for (int w = 1; w < kRowThreads / 32; ++w)
    best = Better(best, ArgBest{sv[w], si[w]});
  return best;
}

__device__ float BlockSum1024(float value, float* shared) {
  value = WaveSum(value);
  const int lane = threadIdx.x % 32, wave = threadIdx.x / 32;
  __syncthreads();
  if (lane == 0)
    shared[wave] = value;
  __syncthreads();
  float total = 0;
  for (int w = 0; w < kRowThreads / 32; ++w)
    total += shared[w];
  return total;
}

__global__ void __launch_bounds__(kRowThreads)
    SampleRowsKernel(const float* logits, bf16_t* probs, float* entropy,
                     std::int32_t* argmax, std::int32_t* sample, int vocab,
                     float softcap, float temperature, std::uint64_t seed,
                     SuppressedTokens suppressed) {
  __shared__ float sv[32];
  __shared__ int si[32];
  const int row = blockIdx.x;
  const float* raw = logits + static_cast<std::size_t>(row) * vocab;
  // z is recomputed in each pass: cheaper than storing it.
  // Suppressed tokens get z = -inf: zero probability everywhere.
  const auto z = [&](int i) {
    for (int j = 0; j < suppressed.count; ++j)
      if (i == suppressed.ids[j])
        return -INFINITY;
    return softcap * tanhf(raw[i] / softcap) / temperature;
  };
  ArgBest best{-FLT_MAX, 1 << 30}, gumbel{-FLT_MAX, 1 << 30};
  float best_rounded = -FLT_MAX;
  for (int i = threadIdx.x; i < vocab; i += kRowThreads) {
    const float value = z(i);
    best = Better(best, ArgBest{value, i});
    best_rounded = fmaxf(best_rounded, RoundBf16(value));
    const float u = (static_cast<float>(Hash(seed, row, i) >> 8) + 0.5F) *
                    (1.0F / 16777216.0F);
    gumbel = Better(gumbel, ArgBest{value - logf(-logf(u)), i});
  }
  best = BlockArgMax(best, sv, si);
  gumbel = BlockArgMax(gumbel, sv, si);
  best_rounded = WaveMax(best_rounded);
  __syncthreads();
  if (threadIdx.x % 32 == 0)
    sv[threadIdx.x / 32] = best_rounded;
  __syncthreads();
  for (int w = 0; w < kRowThreads / 32; ++w)
    best_rounded = fmaxf(best_rounded, sv[w]);
  const float m = best.value;
  float s = 0, u = 0, sb = 0;
  for (int i = threadIdx.x; i < vocab; i += kRowThreads) {
    const float zi = z(i);
    if (zi == -INFINITY)
      continue;  // A suppressed token (e * d would be 0 * -inf).
    const float d = zi - m;
    const float e = expf(d);
    s += e;
    u += e * d;
    sb += expf(RoundBf16(zi) - best_rounded);
  }
  s = BlockSum1024(s, sv);
  u = BlockSum1024(u, sv);
  sb = BlockSum1024(sb, sv);
  for (int i = threadIdx.x; i < vocab; i += kRowThreads) {
    const std::size_t at =
        (static_cast<std::size_t>(i / 32) * gridDim.x + row) * 32 + i % 32;
    probs[at] = ToBf16(expf(RoundBf16(z(i)) - best_rounded) / sb);
  }
  if (threadIdx.x == 0) {
    entropy[row] = logf(s) - u / s;
    argmax[row] = best.index;
    sample[row] = gumbel.index;
  }
}

__global__ void __launch_bounds__(kThreads)
    SoftEmbedNormKernel(const float* soft, const bf16_t* pre_norm, float scale,
                        bf16_t* out, int hidden, float eps) {
  __shared__ float shared[8];
  const std::size_t base = static_cast<std::size_t>(blockIdx.x) * hidden;
  float v[kMaxPerThread];
  float sum = 0;
#pragma unroll
  for (int j = 0; j < kMaxPerThread; ++j) {
    const int i = threadIdx.x + j * kThreads;
    v[j] = i < hidden ? RoundBf16(RoundBf16(soft[base + i]) * scale) : 0.0F;
    sum += v[j] * v[j];
  }
  const float r = RmsScale(BlockSum(sum, shared), hidden, eps);
#pragma unroll
  for (int j = 0; j < kMaxPerThread; ++j) {
    const int i = threadIdx.x + j * kThreads;
    if (i < hidden)
      out[base + i] = ToBf16(v[j] * r * ToFloat(pre_norm[i]));
  }
}

__global__ void __launch_bounds__(kThreads)
    SelfConditionCombineKernel(const float* embeds, const float* signal,
                               float* h, int hidden, float eps) {
  __shared__ float shared[8];
  const std::size_t base = static_cast<std::size_t>(blockIdx.x) * hidden;
  float v[kMaxPerThread];
  float sum = 0;
#pragma unroll
  for (int j = 0; j < kMaxPerThread; ++j) {
    const int i = threadIdx.x + j * kThreads;
    v[j] = 0;
    if (i < hidden)
      v[j] = RoundBf16(embeds[base + i] +
                       (signal ? RoundBf16(signal[base + i]) : 0.0F));
    sum += v[j] * v[j];
  }
  const float r = RmsScale(BlockSum(sum, shared), hidden, eps);
#pragma unroll
  for (int j = 0; j < kMaxPerThread; ++j) {
    const int i = threadIdx.x + j * kThreads;
    if (i < hidden)
      h[base + i] = RoundBf16(v[j] * r);
  }
}

__global__ void SumPartialsKernel(const float4* partials, float4* out,
                                  std::size_t count4, int splits) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count4)
    return;
  float4 sum = partials[i];
  for (int s = 1; s < splits; ++s) {
    const float4 v = partials[s * count4 + i];
    sum.x += v.x;
    sum.y += v.y;
    sum.z += v.z;
    sum.w += v.w;
  }
  out[i] = sum;
}

__global__ void DequantQ8Kernel(const std::uint8_t* blocks, bf16_t* out,
                                std::size_t values) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= values)
    return;
  const std::uint8_t* block = blocks + (i / 32) * 34;
  const float scale = __half2float(*reinterpret_cast<const __half*>(block));
  out[i] = ToBf16(
      scale * static_cast<float>(static_cast<std::int8_t>(block[2 + i % 32])));
}

__global__ void DequantFp8RowsKernel(const std::uint8_t* values,
                                     const bf16_t* scales, bf16_t* out,
                                     std::size_t count, std::size_t cols) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  out[i] =
      ToBf16(ToFloat(Fp8E4M3ToBf16(values[i])) * ToFloat(scales[i / cols]));
}

__global__ void RepackQ8Kernel(const std::uint8_t* blocks, std::int8_t* values,
                               std::uint16_t* scales, std::size_t count) {
  const std::size_t b =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (b >= count)
    return;
  const std::uint8_t* block = blocks + b * 34;
  scales[b] = static_cast<std::uint16_t>(block[0] | (block[1] << 8));
#pragma unroll
  for (int j = 0; j < 32; ++j)
    values[b * 32 + j] = static_cast<std::int8_t>(block[2 + j]);
}

__global__ void F32ToBf16Kernel(const float* in, bf16_t* out,
                                std::size_t count) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count)
    out[i] = ToBf16(in[i]);
}

}  // namespace

void DequantQ8(const std::uint8_t* blocks, bf16_t* out, std::size_t values,
               hipStream_t stream) {
  DequantQ8Kernel<<<(values + 255) / 256, 256, 0, stream>>>(blocks, out,
                                                            values);
  CheckLaunch("DequantQ8");
}

void DequantFp8Rows(const std::uint8_t* values, const bf16_t* scales,
                    bf16_t* out, std::size_t rows, std::size_t cols,
                    hipStream_t stream) {
  const std::size_t count = rows * cols;
  DequantFp8RowsKernel<<<(count + 255) / 256, 256, 0, stream>>>(
      values, scales, out, count, cols);
  CheckLaunch("DequantFp8Rows");
}

void RepackQ8(const std::uint8_t* blocks, std::int8_t* values,
              std::uint16_t* scales, std::size_t block_count,
              hipStream_t stream) {
  RepackQ8Kernel<<<(block_count + 255) / 256, 256, 0, stream>>>(
      blocks, values, scales, block_count);
  CheckLaunch("RepackQ8");
}

void F32ToBf16(const float* in, bf16_t* out, std::size_t count,
               hipStream_t stream) {
  F32ToBf16Kernel<<<(count + 255) / 256, 256, 0, stream>>>(in, out, count);
  CheckLaunch("F32ToBf16");
}

void SumPartials(const float* partials, float* out, std::size_t count,
                 int splits, hipStream_t stream) {
  if (count % 4 != 0)
    throw std::invalid_argument("SumPartials: count must be a multiple of 4");
  const std::size_t count4 = count / 4;
  SumPartialsKernel<<<(count4 + 255) / 256, 256, 0, stream>>>(
      reinterpret_cast<const float4*>(partials), reinterpret_cast<float4*>(out),
      count4, splits);
  CheckLaunch("SumPartials");
}

void Embed(const std::int32_t* ids, const bf16_t* embed, float scale,
           float* out, int tokens, int hidden, hipStream_t stream) {
  EmbedKernel<<<tokens, kThreads, 0, stream>>>(ids, embed, scale, out, hidden);
  CheckLaunch("Embed");
}

void RmsNormToBf16(const float* x, const bf16_t* w, bf16_t* out, int rows,
                   int hidden, float eps, hipStream_t stream) {
  RmsNormKernel<<<rows, kThreads, 0, stream>>>(x, w, out, hidden, eps);
  CheckLaunch("RmsNorm");
}

void PostAttention(float* h, const float* o, const bf16_t* post_attention,
                   const bf16_t* pre_ff, const bf16_t* pre_ff_2,
                   const bf16_t* router_scale, bf16_t* ff, bf16_t* ff2,
                   bf16_t* rout, int rows, int hidden, float eps,
                   hipStream_t stream) {
  if (hidden > kMaxPerThread * kThreads)
    throw std::invalid_argument("PostAttention: hidden too large");
  PostAttentionKernel<<<rows, kThreads, 0, stream>>>(
      h, o, post_attention, pre_ff, pre_ff_2, router_scale, ff, ff2, rout,
      hidden, eps);
  CheckLaunch("PostAttention");
}

void FinishLayer(float* h, const float* mlp, const ExpertOutputs& moe,
                 const bf16_t* post_ff_1, const bf16_t* post_ff_2,
                 const bf16_t* post_ff, float layer_scalar, int rows,
                 int hidden, float eps, hipStream_t stream) {
  FinishLayerKernel<<<rows, kThreads, 0, stream>>>(
      h, mlp, moe, post_ff_1, post_ff_2, post_ff, layer_scalar, hidden, eps);
  CheckLaunch("FinishLayer");
}

void QkvNormRope(const float* qkv, const bf16_t* q_norm, const bf16_t* k_norm,
                 RopeTable rope, bf16_t* q_out, bf16_t* k_cache,
                 bf16_t* v_cache, int tokens, int q_heads, int kv_heads,
                 int head_dim, bool shared_kv, int first_position,
                 int first_slot, int capacity, float eps, hipStream_t stream) {
  const dim3 grid(tokens, q_heads + 2 * kv_heads);
  QkvNormRopeKernel<<<grid, head_dim / 2, 0, stream>>>(
      qkv, q_norm, k_norm, rope.inv_freq, q_out, k_cache, v_cache, q_heads,
      kv_heads, head_dim, shared_kv, first_position, first_slot, capacity, eps);
  CheckLaunch("QkvNormRope");
}

void Route(const float* logits, const bf16_t* per_expert_scale,
           std::int32_t* index, float* weight, int rows, int experts, int k,
           hipStream_t stream) {
  if (experts > 128 || k > 8)
    throw std::invalid_argument("Route: unsupported expert count");
  RouteKernel<<<(rows + 7) / 8, 256, 0, stream>>>(
      logits, per_expert_scale, index, weight, rows, experts, k);
  CheckLaunch("Route");
}

void Dispatch(const std::int32_t* index, std::int32_t* counts,
              std::int32_t* offsets, std::int32_t* source, std::int32_t* slot,
              std::int32_t* groups, std::int32_t* group_count, int rows,
              int experts, int k, hipStream_t stream) {
  if (experts > 256)
    throw std::invalid_argument("Dispatch: unsupported expert count");
  DispatchKernel<<<1, 1024, 0, stream>>>(index, counts, offsets, source, slot,
                                         groups, group_count, rows * k, experts,
                                         k);
  CheckLaunch("Dispatch");
}

void SampleRows(const float* logits, bf16_t* probs, float* entropy,
                std::int32_t* argmax, std::int32_t* sample, int rows, int vocab,
                float softcap, float temperature, std::uint64_t seed,
                SuppressedTokens suppressed, hipStream_t stream) {
  if (suppressed.count < 0 || suppressed.count > SuppressedTokens::kMax)
    throw std::invalid_argument("SampleRows: too many suppressed tokens");
  SampleRowsKernel<<<rows, kRowThreads, 0, stream>>>(
      logits, probs, entropy, argmax, sample, vocab, softcap, temperature, seed,
      suppressed);
  CheckLaunch("SampleRows");
}

void SoftEmbedNorm(const float* soft, const bf16_t* pre_norm, float scale,
                   bf16_t* out, int rows, int hidden, float eps,
                   hipStream_t stream) {
  SoftEmbedNormKernel<<<rows, kThreads, 0, stream>>>(soft, pre_norm, scale, out,
                                                     hidden, eps);
  CheckLaunch("SoftEmbedNorm");
}

void SelfConditionCombine(const float* embeds, const float* signal, float* h,
                          int rows, int hidden, float eps, hipStream_t stream) {
  SelfConditionCombineKernel<<<rows, kThreads, 0, stream>>>(embeds, signal, h,
                                                            hidden, eps);
  CheckLaunch("SelfConditionCombine");
}

void ShiftCache(bf16_t* cache, bf16_t* scratch, int heads, int capacity,
                int head_dim, int from, int count, hipStream_t stream) {
  if (count <= 0 || from == 0)
    return;
  const std::size_t row = static_cast<std::size_t>(head_dim) * sizeof(bf16_t);
  const std::size_t pitch = row * capacity;
  const std::size_t width = row * count;
  auto status =
      hipMemcpy2DAsync(scratch, width, cache + std::size_t(from) * head_dim,
                       pitch, width, heads, hipMemcpyDeviceToDevice, stream);
  if (status == hipSuccess)
    status = hipMemcpy2DAsync(cache, pitch, scratch, width, width, heads,
                              hipMemcpyDeviceToDevice, stream);
  if (status != hipSuccess)
    throw std::runtime_error(std::string("ShiftCache: ") +
                             hipGetErrorString(status));
}

}  // namespace gufo::models::diffusion_gemma::kernels
