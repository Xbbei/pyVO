#include "src/util/math.h"

size_t NChooseK(const size_t n, const size_t k) {
  if (k == 0) {
    return 1;
  }

  return (n * NChooseK(n - 1, k - 1)) / k;
}