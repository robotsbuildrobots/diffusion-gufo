#ifndef GUFO_MODELS_DIFFUSION_GEMMA_ENGINE_HPP_
#define GUFO_MODELS_DIFFUSION_GEMMA_ENGINE_HPP_

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <vector>

#include "src/models/diffusion_gemma/config.hpp"
#include "src/models/diffusion_gemma/tokenizer.hpp"

namespace gufo::models::diffusion_gemma {

struct EngineOptions {
  /// Largest prompt plus generated length; sizes the global-layer KV cache.
  int max_context{131072};
  /// Prompt tokens per encoder pass. Larger chunks give the expert products
  /// fuller row groups; each prompt row costs about 390 KB of scratch and
  /// sliding-layer cache.
  int prefill_chunk{8192};
  /// Print per-phase GPU time for each request to stderr.
  bool profile{false};
};

struct GenerationOptions {
  int max_new_tokens{256};
  /// Fixed-length generation: the end-of-sequence tokens are never sampled,
  /// so the reply runs to max_new_tokens (benchmarks).
  bool suppress_end{false};
  DiffusionDefaults diffusion;
  std::uint64_t seed{0};
  /// Polled before every encoder chunk and denoising step; true stops the
  /// request (GenerationStats::cancelled).
  std::function<bool()> cancelled;
};

struct GenerationStats {
  int prompt_tokens{0};
  int reused_tokens{0};  ///< Prompt prefix reused from the previous cache.
  int generated_tokens{0};
  int blocks{0};
  int denoising_steps{0};
  double prefill_ms{0};      ///< Prompt encoding.
  double denoise_ms{0};      ///< All denoising steps.
  double commit_ms{0};       ///< Encoding accepted canvases into the cache.
  double first_block_ms{0};  ///< Prompt submission to the first canvas.
  bool stopped_by_eos{false};
  bool cancelled{false};
};

/// Single-session DiffusionGemma block-diffusion engine on gfx1151.
class Engine {
public:
  /// Called once per accepted canvas with the tokens it contributes (EOS and
  /// everything after it removed). Return false to cancel.
  using BlockCallback = std::function<bool(std::span<const std::int32_t>)>;

  Engine(const std::filesystem::path& root, const EngineOptions& options);
  ~Engine();
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  [[nodiscard]] const Config& config() const;
  [[nodiscard]] const Tokenizer& tokenizer() const;
  [[nodiscard]] const DiffusionDefaults& diffusion_defaults() const;
  [[nodiscard]] int max_context() const;
  [[nodiscard]] std::size_t weight_bytes() const;

  /// Generates a reply to `prompt`. The longest prefix it shares with the
  /// cached tokens (the previous prompt and its reply's committed canvases)
  /// is reused when the sliding layers still hold the window before it;
  /// only the rest is encoded.
  GenerationStats Generate(std::span<const std::int32_t> prompt,
                           const GenerationOptions& options,
                           const BlockCallback& on_block);

  /// Runs an encoder chunk and two denoising steps so that the first
  /// request does not pay for loading kernels. Leaves the cache empty;
  /// returns elapsed milliseconds.
  double Warmup();

  // Benchmark primitives. Reset() empties the cache; Encode() appends tokens
  // through the causal encoder and returns elapsed milliseconds.
  void Reset();
  double Encode(std::span<const std::int32_t> tokens);
  [[nodiscard]] int cached_tokens() const;

  /// Reference check: encodes `prompt`, runs the denoiser on `canvas`
  /// without self-conditioning, then on `canvas2` conditioned on the first
  /// step's logits (temperature 1). Returns both steps' softcapped logits,
  /// row-major [2][canvas][vocab].
  std::vector<float> DebugLogits(std::span<const std::int32_t> prompt,
                                 const std::vector<std::int32_t>& canvas,
                                 const std::vector<std::int32_t>& canvas2);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::diffusion_gemma

#endif  // GUFO_MODELS_DIFFUSION_GEMMA_ENGINE_HPP_
