#ifndef GUFO_SERVER_DIFFUSION_GEMMA_BACKEND_HPP_
#define GUFO_SERVER_DIFFUSION_GEMMA_BACKEND_HPP_

#include <filesystem>
#include <memory>
#include <string>

#include "src/cli/serve/text_generation_backend.hpp"
#include "src/models/diffusion_gemma/engine.hpp"

namespace gufo::server {

/// HTTP text backend for DiffusionGemma's block diffusion. Up to the
/// engine's session count of requests run at once, their denoising steps
/// batched; output arrives a canvas (256 tokens) at a time. The diffusion
/// sampler replaces token sampling, so temperature/top-k/top-p and penalties
/// do not apply; `seed` makes a request reproducible. Thinking is off unless
/// the request enables it (`chat_template_kwargs.enable_thinking` or
/// `reasoning`); the thinking channel is then returned as reasoning, and
/// otherwise removed. Raw completions keep it as `<think>` text and accept
/// `ignore_eos` (the end tokens are never sampled) and `cache_prompt: false`
/// (no prompt prefix is reused) and `return_token_ids`. Tools are rendered in
/// the checkpoint's native format and its calls are returned to the adapter as
/// Qwen XML call blocks, which it parses into tool calls.
class DiffusionGemmaBackend final : public TextGenerationBackend {
public:
  using TextGenerationBackend::chat;
  using TextGenerationBackend::complete;

  DiffusionGemmaBackend(const std::filesystem::path& root,
                        const models::diffusion_gemma::EngineOptions& options,
                        std::string model_id);

  [[nodiscard]] std::string model_id() const override { return model_id_; }
  [[nodiscard]] bool ready() const override { return true; }
  /// The diffusion sampler cannot apply a JSON grammar token by token.
  [[nodiscard]] bool supports_response_format() const override { return false; }
  [[nodiscard]] std::uint32_t max_context() const override;
  [[nodiscard]] SamplingDefaults sampling_defaults() const override;
  /// A thinking reply starts with `<think>` (the thinking channel).
  [[nodiscard]] InitialOutputState initial_output_state(
      const ChatRequest& request) const override {
    return request.reasoning.enabled.value_or(false)
               ? InitialOutputState::kAuto
               : InitialOutputState::kContent;
  }

  Result complete(std::string_view prompt, std::size_t max_tokens,
                  const sampling::SamplingConfig& sampling,
                  const CancellationCheck& is_cancelled = {},
                  const TokenCallback& on_token = {},
                  std::string_view client_id = "anonymous",
                  const std::vector<std::string>& stop_sequences = {},
                  bool cache_prompt = true) override;

  Result chat(const ChatRequest& request, std::size_t max_tokens,
              const sampling::SamplingConfig& sampling,
              const CancellationCheck& is_cancelled = {},
              const TokenCallback& on_token = {}) override;

  /// Raw completions, with `ignore_eos` (fixed-length benchmarks).
  std::shared_ptr<GenerationRequest> start_complete(
      std::string_view prompt, std::size_t max_tokens,
      const sampling::SamplingConfig& sampling,
      const CancellationCheck& is_cancelled = {}, bool stream_output = false,
      bool ignore_eos = false, std::string_view client_id = "anonymous",
      const std::vector<std::string>& stop_sequences = {},
      bool return_progress = false, bool cache_prompt = true) override;

  [[nodiscard]] std::size_t count_tokens(std::string_view text) const override;

  void set_default_max_tokens(std::size_t tokens) {
    default_max_tokens_ = tokens;
  }
  /// See Engine::Warmup.
  double Warmup() { return engine_->Warmup(); }

private:
  /// How the thinking channel appears in the reply.
  enum class Thinking { kRemove, kTagged };
  Result CompleteRaw(std::string_view prompt, std::size_t max_tokens,
                     const sampling::SamplingConfig& sampling,
                     const CancellationCheck& is_cancelled,
                     const TokenCallback& on_token, std::string_view client_id,
                     const std::vector<std::string>& stop_sequences,
                     bool ignore_eos, bool cache_prompt,
                     const TokenIdsCallback& on_token_ids = {});
  Result Run(const std::vector<std::int32_t>& prompt, std::size_t max_tokens,
             const sampling::SamplingConfig& sampling,
             const CancellationCheck& is_cancelled,
             const TokenCallback& on_token, std::string_view client_id,
             const std::vector<std::string>& stop_sequences, Thinking thinking,
             bool ignore_eos, bool cache_prompt = true,
             const TokenIdsCallback& on_token_ids = {});

  std::unique_ptr<models::diffusion_gemma::Engine> engine_;
  std::string model_id_;
  std::size_t default_max_tokens_{0};
};

}  // namespace gufo::server

#endif  // GUFO_SERVER_DIFFUSION_GEMMA_BACKEND_HPP_
