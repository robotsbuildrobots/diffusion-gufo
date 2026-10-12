#include "src/models/diffusion_gemma/engine.hpp"

#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>

#include "src/models/diffusion_gemma/cache_state.hpp"
#include "src/models/diffusion_gemma/denoise.hpp"
#include "src/models/diffusion_gemma/hip_check.hpp"
#include "src/models/diffusion_gemma/kernels.hpp"
#include "src/models/diffusion_gemma/profiler.hpp"
#include "src/models/diffusion_gemma/weights.hpp"

namespace gufo::models::diffusion_gemma {
namespace {

/// Owning device allocation, zero-initialised.
class Buffer {
public:
  Buffer() = default;
  explicit Buffer(std::size_t bytes) : bytes_(bytes) {
    HipCheck(hipMalloc(&data_, bytes), "allocation");
    // The memset runs on the null stream, which the engine's non-blocking
    // stream does not wait for: finish it before the buffer is used.
    HipCheck(hipMemset(data_, 0, bytes), "memset");
    HipCheck(hipStreamSynchronize(nullptr), "memset");
  }
  ~Buffer() {
    if (data_)
      (void)hipFree(data_);
  }
  Buffer(Buffer&& other) noexcept : data_(other.data_), bytes_(other.bytes_) {
    other.data_ = nullptr;
  }
  Buffer& operator=(Buffer&& other) noexcept {
    std::swap(data_, other.data_);
    std::swap(bytes_, other.bytes_);
    return *this;
  }
  template<typename T>
  T* as() const {
    return static_cast<T*>(data_);
  }
  [[nodiscard]] std::size_t bytes() const { return bytes_; }

private:
  void* data_{nullptr};
  std::size_t bytes_{0};
};

using Clock = std::chrono::steady_clock;
double Ms(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

}  // namespace

struct Engine::Impl {
  // Most key-range splits of a canvas attention pass (merged afterwards).
  static constexpr int kMaxSplits = 16;
  // Vocabulary splits of the self-conditioning product: 16 x 22 column
  // tiles = 352 workgroups, enough to fill the GPU (see soft_embed.hip).
  static constexpr int kSoftEmbedSplits = 16;
  // Sliding positions kept beyond the attention window, so that a prompt
  // diverging from the cache up to this far before its end (an edited or
  // re-tokenised reply) still reuses the cache (CacheState::Reuse).
  static constexpr int kReuseHistory = 4096;
  // Per-block seed offset: an odd constant, so blocks of one request draw
  // from distinct streams.
  static constexpr std::uint64_t kBlockSeedStride = 0x1000193ULL;

  std::filesystem::path root;
  EngineOptions options;
  Config config;
  DiffusionDefaults defaults;
  std::unique_ptr<Tokenizer> tokenizer;
  std::unique_ptr<DeviceWeights> device_weights;
  hipStream_t stream{};
  std::unique_ptr<Profiler> profiler;

  int rows{0};  // Scratch rows: max(prefill chunk, canvas).
  float embed_scale{0};

  // Layer pass scratch, `rows` rows (an encoder chunk or the canvas):
  // residual stream, normed input, attention and dense MLP intermediates.
  Buffer ids, h, xn, qkv, q, attn, o, ff, ff2, rout, act, mlp_out;
  // Experts: router logits, top-k choice, the dispatch grouping rows by
  // expert (and its list of 64-row groups), per-assignment expert
  // intermediates and outputs.
  Buffer router_logits, top_index, top_weight, counts, offsets, source, slot,
      groups, group_count, expert_act, expert_out;
  // Canvas attention split-key partials.
  Buffer split_out, split_stats;
  // Denoiser head and sampler, canvas rows: logits, self-conditioning
  // probabilities (blocked for SoftEmbedProduct), per-row entropy, argmax
  // and sample.
  Buffer logits, probs, entropy, argmax, sample;
  // Self-conditioning: token embeddings, soft embedding (and its split
  // partials), normed soft embedding, the conditioning signal.
  Buffer embeds, soft, soft_partials, soft_norm, signal;

  // KV caches, per layer, [kv heads][capacity][head dim]; see CacheState
  // for the slot layout. `shift_scratch` moves the sliding caches' kept
  // positions to the front after each encoder pass.
  CacheState cache{0, kReuseHistory};
  int global_capacity{0};
  int sliding_capacity{0};
  std::vector<Buffer> k_cache, v_cache;
  Buffer shift_scratch;
  Buffer sliding_freq, global_freq;  // RoPE inverse frequencies.

  // The LM head is the tied embedding, which stays row-major for lookups
  // and the self-conditioning product: kernels::Linear reads this blocked
  // copy. `linear_partials` holds split-reduction partials.
  Buffer lm_head, linear_partials;

  // Profile mode: experts with at least one row, over canvas passes.
  std::size_t active_experts{0}, expert_passes{0};

  Impl(const std::filesystem::path& r, const EngineOptions& o)
      : root(r), options(o) {
    config = LoadConfig(root);
    defaults = LoadDiffusionDefaults(root);
    tokenizer = std::make_unique<Tokenizer>(root);
    cache = CacheState(config.sliding_window, kReuseHistory);
    options.prefill_chunk =
        std::max(256, (options.prefill_chunk + 63) / 64 * 64);
    options.max_context = std::clamp(options.max_context, config.canvas + 1,
                                     config.max_positions);
    HipCheck(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "stream");
    device_weights = std::make_unique<DeviceWeights>(root, config);
    profiler = std::make_unique<Profiler>(stream, options.profile);
    // The embedding scale is a checkpoint-dtype (BF16) constant.
    {
      const float exact = std::sqrt(static_cast<float>(config.hidden));
      std::uint32_t bits;
      std::memcpy(&bits, &exact, 4);
      bits = (bits + 0x7fffU + ((bits >> 16) & 1U)) & 0xffff0000U;
      std::memcpy(&embed_scale, &bits, 4);
    }
    Allocate();
  }

  ~Impl() {
    if (stream) {
      (void)hipStreamSynchronize(stream);
      profiler.reset();
      (void)hipStreamDestroy(stream);
    }
  }

  void Allocate() {
    const Config& c = config;
    rows = std::max(options.prefill_chunk, c.canvas);
    const std::size_t R = rows, H = c.hidden, V = c.vocab;
    const std::size_t qkv_width =
        static_cast<std::size_t>(c.heads) * c.global_head_dim +
        static_cast<std::size_t>(c.global_kv_heads) * c.global_head_dim;
    const std::size_t qkv_width_sliding =
        static_cast<std::size_t>(c.heads + 2 * c.sliding_kv_heads) *
        c.sliding_head_dim;
    const std::size_t attn_width =
        static_cast<std::size_t>(c.heads) * c.global_head_dim;
    const std::size_t A = R * c.top_k;
    ids = Buffer(R * 4);
    h = Buffer(R * H * 4);
    embeds = Buffer(c.canvas * H * 4);
    xn = Buffer(R * H * 2);
    qkv = Buffer(R * std::max(qkv_width, qkv_width_sliding) * 4);
    q = Buffer(R * attn_width * 2);
    attn = Buffer(R * attn_width * 2);
    o = Buffer(R * H * 4);
    ff = Buffer(R * H * 2);
    ff2 = Buffer(R * H * 2);
    rout = Buffer(R * H * 2);
    act = Buffer(R * c.intermediate * 2);
    mlp_out = Buffer(R * H * 4);
    router_logits = Buffer(R * c.experts * 4);
    top_index = Buffer(A * 4);
    top_weight = Buffer(A * 4);
    counts = Buffer(c.experts * 4);
    offsets = Buffer(c.experts * 4);
    source = Buffer(A * 4);
    slot = Buffer(A * 4);
    groups = Buffer(
        std::size_t(kernels::ExpertGroupBound(rows, c.experts, c.top_k, 64)) *
        8);
    group_count = Buffer(4);
    expert_act = Buffer(A * c.expert_intermediate * 2);
    expert_out = Buffer(A * H * 2);
    logits = Buffer(c.canvas * V * 4);
    probs = Buffer(c.canvas * V * 2);
    soft = Buffer(c.canvas * H * 4);
    soft_partials = Buffer(std::size_t(kSoftEmbedSplits) * c.canvas * H * 4);
    soft_norm = Buffer(c.canvas * H * 2);
    signal = Buffer(c.canvas * H * 4);
    entropy = Buffer(c.canvas * 4);
    argmax = Buffer(c.canvas * 4);
    sample = Buffer(c.canvas * 4);
    // Split-key attention partials for canvas passes.
    split_out = Buffer(std::size_t(kMaxSplits) * c.canvas * attn_width * 4);
    split_stats = Buffer(std::size_t(kMaxSplits) * c.canvas * c.heads * 2 * 4);

    // Global layers hold every position; sliding layers the window, the
    // reuse history and one pass (an encoder chunk or a canvas).
    global_capacity = options.max_context + c.canvas;
    sliding_capacity = cache.sliding_keep() + rows;
    shift_scratch = Buffer(static_cast<std::size_t>(c.sliding_kv_heads) *
                           cache.sliding_keep() * c.sliding_head_dim * 2);
    for (int l = 0; l < c.layers; ++l) {
      const std::size_t capacity =
          c.is_global[l] ? global_capacity : sliding_capacity;
      const std::size_t bytes = static_cast<std::size_t>(c.kv_heads(l)) *
                                capacity * c.head_dim(l) * 2;
      k_cache.emplace_back(bytes);
      v_cache.emplace_back(bytes);
    }

    // RoPE inverse frequencies, as transformers computes them in FP32.
    std::vector<float> sliding(c.sliding_head_dim / 2),
        global(c.global_head_dim / 2, 0.0F);
    for (int i = 0; i < c.sliding_head_dim / 2; ++i)
      sliding[i] = 1.0F / std::pow(c.sliding_theta,
                                   static_cast<float>(2 * i) /
                                       static_cast<float>(c.sliding_head_dim));
    const int rotated =
        static_cast<int>(c.global_rotary_fraction * c.global_head_dim / 2);
    for (int i = 0; i < rotated; ++i)
      global[i] = 1.0F / std::pow(c.global_theta,
                                  static_cast<float>(2 * i) /
                                      static_cast<float>(c.global_head_dim));
    sliding_freq = Buffer(sliding.size() * 4);
    global_freq = Buffer(global.size() * 4);
    HipCheck(hipMemcpy(sliding_freq.as<void>(), sliding.data(),
                       sliding.size() * 4, hipMemcpyHostToDevice),
             "rope upload");
    HipCheck(hipMemcpy(global_freq.as<void>(), global.data(), global.size() * 4,
                       hipMemcpyHostToDevice),
             "rope upload");

    lm_head = Buffer(V * H * 2);
    kernels::BlockLinearWeights(device_weights->view().embed,
                                lm_head.as<bf16_t>(), c.vocab, c.hidden,
                                stream);
    // Partials for the largest split reduction of any pass's products.
    std::size_t partials = kernels::LinearPartialFloats(c.canvas, c.vocab, H);
    const std::pair<int, int> shapes[] = {
        {static_cast<int>(qkv_width), c.hidden},
        {static_cast<int>(qkv_width_sliding), c.hidden},
        {c.hidden, static_cast<int>(attn_width)},
        {c.hidden, c.heads * c.sliding_head_dim},
        {2 * c.intermediate, c.hidden},
        {c.hidden, c.intermediate},
        {c.experts, c.hidden}};
    for (int m = 1; m <= rows; ++m)
      for (const auto& [n, k] : shapes)
        partials = std::max(partials, kernels::LinearPartialFloats(m, n, k));
    if (partials > 0)
      linear_partials = Buffer(partials * 4);
    HipCheck(hipStreamSynchronize(stream), "blocking the LM head");
  }

  /// out (fp32) = a (row-major bf16, `t` rows) * W^T, W blocked.
  void Dense(const void* a, const bf16_t* w, float* out, int t, int n, int k) {
    kernels::Linear(static_cast<const bf16_t*>(a), w,
                    linear_partials.as<float>(), out, t, n, k, stream);
  }

  /// Input norm, QKV, Q/K norms and RoPE (writing this pass's keys and
  /// values to the cache), attention and the output projection; leaves the
  /// three feed-forward inputs (ff, ff2, rout).
  void AttentionBlock(int l, int t, bool decoder) {
    const Config& c = config;
    const auto& w = device_weights->view().layers[l];
    const bool global = c.is_global[l];
    const int H = c.hidden, hd = c.head_dim(l), kv = c.kv_heads(l);
    const int qkv_rows = (c.heads + (global ? 1 : 2) * kv) * hd;
    profiler->Phase(global ? "qkv_global" : "qkv_sliding");
    kernels::RmsNormToBf16(h.as<float>(), w.input_norm, xn.as<bf16_t>(), t, H,
                           c.rms_eps, stream);
    Dense(xn.as<void>(), w.qkv, qkv.as<float>(), t, qkv_rows, H);
    const int capacity = global ? global_capacity : sliding_capacity;
    const auto pass = global ? cache.GlobalPass() : cache.SlidingPass();
    const std::size_t skip_elements = static_cast<std::size_t>(pass.skip) * hd;
    kernels::QkvNormRope(qkv.as<float>(), w.q_norm, w.k_norm,
                         kernels::RopeTable{global ? global_freq.as<float>()
                                                   : sliding_freq.as<float>()},
                         q.as<bf16_t>(), k_cache[l].as<bf16_t>(),
                         v_cache[l].as<bf16_t>(), t, c.heads, kv, hd, global,
                         cache.length(), pass.first_slot, capacity, c.rms_eps,
                         stream);
    // Encoder passes are causal (sliding layers also windowed); decoder
    // passes see the cache and the whole canvas.
    profiler->Phase(global ? "attn_global" : "attn_sliding");
    kernels::Attention(
        q.as<bf16_t>(), k_cache[l].as<bf16_t>() + skip_elements,
        v_cache[l].as<bf16_t>() + skip_elements, attn.as<bf16_t>(), t, c.heads,
        kv, hd, capacity, pass.first_slot + t - pass.skip, cache.length(),
        pass.first_position, /*causal=*/!decoder,
        (!decoder && !global) ? c.sliding_window : 0, stream,
        kernels::SplitWorkspace{.out = split_out.as<float>(),
                                .stats = split_stats.as<float>(),
                                .capacity = split_out.bytes() / 4,
                                .max_splits = kMaxSplits});
    profiler->Phase(global ? "o_global" : "o_sliding");
    Dense(attn.as<void>(), w.o, o.as<float>(), t, H, c.heads * hd);
    kernels::PostAttention(h.as<float>(), o.as<float>(), w.post_attention_norm,
                           w.pre_ff_norm, w.pre_ff_norm_2, w.router_scale,
                           ff.as<bf16_t>(), ff2.as<bf16_t>(), rout.as<bf16_t>(),
                           t, H, c.rms_eps, stream);
  }

  /// Dense MLP on `ff` into `mlp_out`.
  void DenseMlpBlock(int l, int t) {
    const Config& c = config;
    const auto& w = device_weights->view().layers[l];
    profiler->Phase("dense_mlp");
    kernels::LinearGeluMul(ff.as<bf16_t>(), w.mlp_gate_up, act.as<bf16_t>(), t,
                           c.intermediate, c.hidden, stream);
    Dense(act.as<void>(), w.mlp_down, mlp_out.as<float>(), t, c.hidden,
          c.intermediate);
  }

  /// Rows per expert group for a pass of `t` rows: 128 for prompt chunks,
  /// whose experts average about 2t*8/128 rows, so that each widened weight
  /// tile serves twice the rows; 64 for canvases (about 25 rows per expert)
  /// and for Q8_0, whose per-block scaling makes the larger tile slower.
  int ExpertGroupRows(int t) const {
    const bool q8 =
        device_weights->view().layers[0].expert_gate_up_q8 != nullptr;
    return t > config.canvas && !q8 ? 128 : 64;
  }

  /// Mixture of experts on `ff2` (routed by `rout`): per-assignment outputs
  /// in `expert_out`, combined by FinishLayer.
  void ExpertBlock(int l, int t) {
    const Config& c = config;
    const auto& w = device_weights->view().layers[l];
    const int H = c.hidden;
    profiler->Phase("router");
    Dense(rout.as<void>(), w.router_proj, router_logits.as<float>(), t,
          c.experts, H);
    kernels::Route(router_logits.as<float>(), w.per_expert_scale,
                   top_index.as<std::int32_t>(), top_weight.as<float>(), t,
                   c.experts, c.top_k, stream);
    const int group_rows = ExpertGroupRows(t);
    kernels::Dispatch(top_index.as<std::int32_t>(), counts.as<std::int32_t>(),
                      offsets.as<std::int32_t>(), source.as<std::int32_t>(),
                      slot.as<std::int32_t>(), groups.as<std::int32_t>(),
                      group_count.as<std::int32_t>(), t, c.experts, c.top_k,
                      group_rows, stream);
    const kernels::ExpertRouting routing{
        .counts = counts.as<std::int32_t>(),
        .offsets = offsets.as<std::int32_t>(),
        .source = source.as<std::int32_t>(),
        .groups = groups.as<std::int32_t>(),
        .group_count = group_count.as<std::int32_t>(),
        .max_groups =
            kernels::ExpertGroupBound(t, c.experts, c.top_k, group_rows),
        .group_rows = group_rows};
    if (profiler->enabled() && t == c.canvas)
      CountActiveExperts();
    profiler->Phase("expert_gate_up");
    kernels::ExpertGateUp(
        ff2.as<bf16_t>(), w.expert_gate_up, routing, expert_act.as<bf16_t>(), H,
        c.expert_intermediate, stream,
        kernels::ExpertQ8{w.expert_gate_up_q8, w.expert_gate_up_scale},
        kernels::ExpertFp8{w.expert_gate_up_fp8, w.expert_gate_up_fp8_scale},
        kernels::ExpertFp4{w.expert_gate_up_fp4,
                           w.expert_gate_up_fp4_block_scale,
                           w.expert_gate_up_fp4_tensor_scale});
    profiler->Phase("expert_down");
    kernels::ExpertDown(
        expert_act.as<bf16_t>(), w.expert_down, routing,
        expert_out.as<bf16_t>(), H, c.expert_intermediate, stream,
        kernels::ExpertQ8{w.expert_down_q8, w.expert_down_scale},
        kernels::ExpertFp8{w.expert_down_fp8, w.expert_down_fp8_scale},
        kernels::ExpertFp4{w.expert_down_fp4, w.expert_down_fp4_block_scale,
                           w.expert_down_fp4_tensor_scale});
  }

  /// Profile mode: experts with at least one row in this canvas pass.
  void CountActiveExperts() {
    std::vector<std::int32_t> host(config.experts);
    HipCheck(hipMemcpyAsync(host.data(), counts.as<void>(), host.size() * 4,
                            hipMemcpyDeviceToHost, stream),
             "profile counts");
    HipCheck(hipStreamSynchronize(stream), "profile counts");
    for (const auto n : host)
      active_experts += n > 0 ? 1 : 0;
    ++expert_passes;
  }

  /// Runs every layer over the `t` rows in `h`. Encoder passes are causal
  /// and write the cache; decoder passes see the cache plus the canvas.
  void Layers(int t, bool decoder) {
    const Config& c = config;
    for (int l = 0; l < c.layers; ++l) {
      const auto& w = device_weights->view().layers[l];
      AttentionBlock(l, t, decoder);
      DenseMlpBlock(l, t);
      ExpertBlock(l, t);
      profiler->Phase("finish");
      kernels::FinishLayer(
          h.as<float>(), mlp_out.as<float>(),
          kernels::ExpertOutputs{.expert_out = expert_out.as<bf16_t>(),
                                 .slot = slot.as<std::int32_t>(),
                                 .weight = top_weight.as<float>(),
                                 .k = c.top_k},
          w.post_ff_norm_1, w.post_ff_norm_2, w.post_ff_norm, w.layer_scalar, t,
          c.hidden, c.rms_eps, stream);
    }
  }

  void EncodeChunk(const std::int32_t* tokens, int t) {
    const Config& c = config;
    HipCheck(hipMemcpyAsync(ids.as<void>(), tokens, t * 4,
                            hipMemcpyHostToDevice, stream),
             "token upload");
    profiler->Phase("encode_embed");
    kernels::Embed(ids.as<std::int32_t>(), device_weights->view().embed,
                   embed_scale, h.as<float>(), t, c.hidden, stream);
    Layers(t, /*decoder=*/false);
    profiler->Phase("cache_shift");
    // Keep the newest positions at the front of the sliding caches.
    const auto shift = cache.Append(std::span<const std::int32_t>(tokens, t));
    for (int l = 0; l < c.layers; ++l) {
      if (c.is_global[l])
        continue;
      for (auto* buffer : {&k_cache[l], &v_cache[l]})
        kernels::ShiftCache(buffer->as<bf16_t>(), shift_scratch.as<bf16_t>(),
                            c.sliding_kv_heads, sliding_capacity,
                            c.sliding_head_dim, shift.from, shift.count,
                            stream);
    }
  }

  double Encode(std::span<const std::int32_t> tokens,
                const std::function<bool()>& cancelled = {}) {
    if (cache.length() + static_cast<int>(tokens.size()) > options.max_context)
      throw std::length_error("context exceeds the configured context");
    const auto start = Clock::now();
    for (std::size_t done = 0; done < tokens.size();) {
      if (cancelled && cancelled())
        break;
      const int t = static_cast<int>(
          std::min<std::size_t>(options.prefill_chunk, tokens.size() - done));
      EncodeChunk(tokens.data() + done, t);
      done += t;
    }
    HipCheck(hipStreamSynchronize(stream), "encode");
    return Ms(start, Clock::now());
  }

  /// One denoiser forward over `canvas`; leaves per-row entropy, argmax and
  /// a sample in the host vectors and self-conditioning probabilities on the
  /// device.
  void DenoiseStep(const std::vector<std::int32_t>& canvas, bool self_condition,
                   float temperature, std::uint64_t seed,
                   const kernels::SuppressedTokens& suppressed,
                   std::vector<float>* row_entropy,
                   std::vector<std::int32_t>* row_argmax,
                   std::vector<std::int32_t>* row_sample) {
    const Config& c = config;
    const auto& weights = device_weights->view();
    const int n = c.canvas, H = c.hidden;
    HipCheck(hipMemcpyAsync(ids.as<void>(), canvas.data(), n * 4,
                            hipMemcpyHostToDevice, stream),
             "canvas upload");
    profiler->Phase("self_condition");
    kernels::Embed(ids.as<std::int32_t>(), weights.embed, embed_scale,
                   embeds.as<float>(), n, H, stream);
    if (self_condition) {
      kernels::SoftEmbedProduct(probs.as<bf16_t>(), weights.embed,
                                soft_partials.as<float>(), soft.as<float>(), n,
                                c.vocab, H, kSoftEmbedSplits, stream);
      kernels::SoftEmbedNorm(soft.as<float>(), weights.sc_pre_norm, embed_scale,
                             soft_norm.as<bf16_t>(), n, H, c.rms_eps, stream);
      kernels::LinearGeluMul(soft_norm.as<bf16_t>(), weights.sc_gate_up,
                             act.as<bf16_t>(), n, c.intermediate, H, stream);
      Dense(act.as<void>(), weights.sc_down, signal.as<float>(), n, H,
            c.intermediate);
    }
    kernels::SelfConditionCombine(embeds.as<float>(),
                                  self_condition ? signal.as<float>() : nullptr,
                                  h.as<float>(), n, H, c.rms_eps, stream);
    Layers(n, /*decoder=*/true);
    profiler->Phase("lm_head");
    kernels::RmsNormToBf16(h.as<float>(), weights.final_norm, xn.as<bf16_t>(),
                           n, H, c.rms_eps, stream);
    Dense(xn.as<void>(), lm_head.as<bf16_t>(), logits.as<float>(), n, c.vocab,
          H);
    profiler->Phase("sampler");
    kernels::SampleRows(logits.as<float>(), probs.as<bf16_t>(),
                        entropy.as<float>(), argmax.as<std::int32_t>(),
                        sample.as<std::int32_t>(), n, c.vocab, c.final_softcap,
                        temperature, seed, suppressed, stream);
    profiler->Phase("readback");
    row_entropy->resize(n);
    row_argmax->resize(n);
    row_sample->resize(n);
    HipCheck(hipMemcpyAsync(row_entropy->data(), entropy.as<void>(), n * 4,
                            hipMemcpyDeviceToHost, stream),
             "readback");
    HipCheck(hipMemcpyAsync(row_argmax->data(), argmax.as<void>(), n * 4,
                            hipMemcpyDeviceToHost, stream),
             "readback");
    HipCheck(hipMemcpyAsync(row_sample->data(), sample.as<void>(), n * 4,
                            hipMemcpyDeviceToHost, stream),
             "readback");
    HipCheck(hipStreamSynchronize(stream), "denoise step");
  }

  /// Denoises one canvas after the cached context. Returns the final argmax
  /// canvas and the number of denoiser forwards; empty when `cancelled`
  /// fires.
  std::vector<std::int32_t> DenoiseBlock(
      const DiffusionDefaults& d, std::mt19937_64& rng, std::uint64_t seed,
      const std::function<bool()>& cancelled,
      const kernels::SuppressedTokens& suppressed, int* steps) {
    const int n = config.canvas;
    std::uniform_int_distribution<std::int32_t> uniform(0, config.vocab - 1);
    // The canvas starts as random tokens; rows not accepted at a step are
    // drawn again.
    std::vector<std::int32_t> current(n), argmax_canvas, previous_argmax;
    for (auto& token : current)
      token = uniform(rng);
    std::vector<float> row_entropy;
    std::vector<std::int32_t> row_argmax, row_sample;
    *steps = 0;
    for (int step = d.max_denoising_steps; step >= 1; --step) {
      if (cancelled && cancelled())
        return {};
      // Each step draws its Gumbel noise from its own stream.
      const std::uint64_t step_seed =
          MixSeed(seed ^ (static_cast<std::uint64_t>(*steps) << 40));
      DenoiseStep(current, *steps > 0, StepTemperature(d, step), step_seed,
                  suppressed, &row_entropy, &row_argmax, &row_sample);
      ++*steps;
      const auto accepted = AcceptRows(row_entropy, d.entropy_bound);
      for (int i = 0; i < n; ++i)
        current[i] = accepted[i] ? row_sample[i] : uniform(rng);
      argmax_canvas = row_argmax;
      const bool done =
          BlockConverged(d, previous_argmax, row_argmax, row_entropy);
      previous_argmax = row_argmax;
      if (done)
        break;
    }
    return argmax_canvas;
  }

  GenerationStats Generate(std::span<const std::int32_t> prompt,
                           const GenerationOptions& request,
                           const BlockCallback& on_block) {
    const Config& c = config;
    GenerationStats stats;
    stats.prompt_tokens = static_cast<int>(prompt.size());
    if (prompt.empty())
      throw std::invalid_argument("empty prompt");
    if (static_cast<int>(prompt.size()) + c.canvas > options.max_context)
      throw std::length_error(
          "prompt leaves no room for a canvas within the configured context");
    const auto start = Clock::now();
    stats.reused_tokens = cache.Reuse(prompt);
    stats.prefill_ms =
        Encode(prompt.subspan(stats.reused_tokens), request.cancelled);
    profiler->Report("prefill");
    if (request.cancelled && request.cancelled()) {
      stats.cancelled = true;
      return stats;
    }
    std::mt19937_64 rng(MixSeed(request.seed));
    kernels::SuppressedTokens suppressed;
    if (request.suppress_end) {
      if (c.eos_ids.size() > kernels::SuppressedTokens::kMax)
        throw std::invalid_argument("too many end-of-sequence tokens");
      for (const int id : c.eos_ids)
        suppressed.ids[suppressed.count++] = id;
    }
    const int blocks =
        std::max(1, (request.max_new_tokens + c.canvas - 1) / c.canvas);
    for (int b = 0; b < blocks; ++b) {
      if (cache.length() + c.canvas > options.max_context)
        break;
      int steps = 0;
      const auto denoise_start = Clock::now();
      auto canvas =
          DenoiseBlock(request.diffusion, rng,
                       MixSeed(request.seed + kBlockSeedStride * (b + 1)),
                       request.cancelled, suppressed, &steps);
      stats.denoise_ms += Ms(denoise_start, Clock::now());
      if (canvas.empty()) {
        stats.cancelled = true;
        break;
      }
      stats.denoising_steps += steps;
      ++stats.blocks;
      const auto reply = TakeReply(
          canvas, c.eos_ids, request.max_new_tokens - stats.generated_tokens);
      stats.stopped_by_eos = reply.end_of_sequence;
      const int keep = reply.tokens;
      stats.generated_tokens += keep;
      if (b == 0)
        stats.first_block_ms = Ms(start, Clock::now());
      if (on_block &&
          !on_block(std::span<const std::int32_t>(canvas.data(), keep))) {
        stats.cancelled = true;
        break;
      }
      if (stats.stopped_by_eos ||
          stats.generated_tokens >= request.max_new_tokens || b + 1 == blocks)
        break;
      // Reached only when the whole canvas belongs to the reply.
      stats.commit_ms += Encode(canvas);
    }
    if (profiler->enabled() && expert_passes > 0)
      std::fprintf(stderr, "profile active experts per canvas layer: %.1f\n",
                   static_cast<double>(active_experts) / expert_passes);
    profiler->Report("denoise and commit");
    return stats;
  }
};

Engine::Engine(const std::filesystem::path& root, const EngineOptions& options)
    : impl_(std::make_unique<Impl>(root, options)) {}
Engine::~Engine() = default;

const Config& Engine::config() const {
  return impl_->config;
}
const Tokenizer& Engine::tokenizer() const {
  return *impl_->tokenizer;
}
const DiffusionDefaults& Engine::diffusion_defaults() const {
  return impl_->defaults;
}
int Engine::max_context() const {
  return impl_->options.max_context;
}
std::size_t Engine::weight_bytes() const {
  return impl_->device_weights->view().device_bytes;
}

GenerationStats Engine::Generate(std::span<const std::int32_t> prompt,
                                 const GenerationOptions& options,
                                 const BlockCallback& on_block) {
  return impl_->Generate(prompt, options, on_block);
}

double Engine::Warmup() {
  auto& impl = *impl_;
  const auto start = Clock::now();
  const int filler = impl.tokenizer->TokenId("\xE2\x96\x81the");
  std::vector<std::int32_t> tokens(impl.options.prefill_chunk,
                                   filler >= 0 ? filler : 1);
  tokens[0] = impl.config.bos_id;
  // An encoder chunk and two denoising steps (with and without
  // self-conditioning, then a commit) load every kernel before the first
  // request.
  Reset();
  (void)impl.Encode(tokens);
  GenerationOptions options;
  options.diffusion = impl.defaults;
  options.diffusion.max_denoising_steps = 2;
  options.max_new_tokens = 2 * impl.config.canvas;
  (void)impl.Generate(std::span<const std::int32_t>(tokens.data(), 64), options,
                      {});
  Reset();
  return Ms(start, Clock::now());
}

void Engine::Reset() {
  impl_->cache.Clear();
}

double Engine::Encode(std::span<const std::int32_t> tokens) {
  const double ms = impl_->Encode(tokens);
  impl_->profiler->Report();
  return ms;
}

int Engine::cached_tokens() const {
  return impl_->cache.length();
}

std::vector<float> Engine::DebugLogits(
    std::span<const std::int32_t> prompt,
    const std::vector<std::int32_t>& canvas,
    const std::vector<std::int32_t>& canvas2) {
  auto& impl = *impl_;
  const std::size_t rows = impl.config.canvas, vocab = impl.config.vocab;
  Reset();
  (void)impl.Encode(prompt);
  std::vector<float> out(2 * rows * vocab);
  std::vector<float> e;
  std::vector<std::int32_t> a, b;
  const float cap = impl.config.final_softcap;
  for (int step = 0; step < 2; ++step) {
    impl.DenoiseStep(step == 0 ? canvas : canvas2, step == 1, 1.0F, step + 1,
                     {}, &e, &a, &b);
    float* dst = out.data() + step * rows * vocab;
    HipCheck(hipMemcpy(dst, impl.logits.as<void>(), rows * vocab * 4,
                       hipMemcpyDeviceToHost),
             "debug readback");
    for (std::size_t i = 0; i < rows * vocab; ++i)
      dst[i] = cap * std::tanh(dst[i] / cap);
  }
  return out;
}

}  // namespace gufo::models::diffusion_gemma
