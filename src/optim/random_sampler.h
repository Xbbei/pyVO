#pragma once

#include <cstddef>
#include <vector>

#include "src/util/logging.h"

// Random sampler for RANSAC-based methods.
//
// Note that a separate sampler should be instantiated per thread.
class RandomSampler {
 public:
  explicit RandomSampler(const size_t num_samples);

  void Initialize(const size_t total_num_samples);

  size_t MaxNumSamples();

  std::vector<size_t> Sample();

 private:
  const size_t num_samples_;
  std::vector<size_t> sample_idxs_;
};
