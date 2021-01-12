#include "src/util/random.h"

thread_local std::mt19937* PRNG = nullptr;

void SetPRNGSeed(unsigned seed) {
  // Avoid race conditions, especially for srand().
  static std::mutex mutex;
  std::unique_lock<std::mutex> lock(mutex);

  // Overwrite existing PRNG
  if (PRNG != nullptr) {
    delete PRNG;
  }

  PRNG = new std::mt19937(seed);
  srand(seed);
}