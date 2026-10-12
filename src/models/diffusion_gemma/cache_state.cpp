#include "src/models/diffusion_gemma/cache_state.hpp"

#include <algorithm>

namespace gufo::models::diffusion_gemma {

CacheState::CacheState(int sliding_window, int reuse_history)
    : sliding_window_(sliding_window), reuse_history_(reuse_history) {}

CacheState::Pass CacheState::GlobalPass() const {
  return {.first_slot = length_, .skip = 0, .first_position = 0};
}

CacheState::Pass CacheState::SlidingPass() const {
  // Only the newest window - 1 cached positions are attended to.
  const int skip =
      sliding_valid_ - std::min(sliding_valid_, sliding_window_ - 1);
  return {.first_slot = sliding_valid_,
          .skip = skip,
          .first_position = length_ - sliding_valid_ + skip};
}

int CacheState::Reuse(std::span<const std::int32_t> prompt) {
  const int limit = std::min<int>(length_, static_cast<int>(prompt.size()));
  int reuse = 0;
  while (reuse < limit && tokens_[reuse] == prompt[reuse])
    ++reuse;
  // The sliding layers continue at `reuse` only if they still hold the
  // window before it.
  const int window_start = reuse - std::min(reuse, sliding_window_ - 1);
  if (reuse == 0 || window_start < length_ - sliding_valid_) {
    Clear();
    return 0;
  }
  sliding_valid_ -= length_ - reuse;
  length_ = reuse;
  tokens_.resize(reuse);
  return reuse;
}

CacheState::Shift CacheState::Append(std::span<const std::int32_t> tokens) {
  const int t = static_cast<int>(tokens.size());
  const int filled = sliding_valid_ + t;
  const int keep = std::min(filled, sliding_keep());
  length_ += t;
  sliding_valid_ = keep;
  tokens_.insert(tokens_.end(), tokens.begin(), tokens.end());
  return {.from = filled - keep, .count = keep};
}

void CacheState::Clear() {
  length_ = 0;
  sliding_valid_ = 0;
  tokens_.clear();
}

}  // namespace gufo::models::diffusion_gemma
