#pragma once

#include <cstddef>
#include <limits>
#include <vector>

struct Support {
  // The number of inliers.
  size_t num_inliers = 0;

  // The sum of all inlier residuals.
  double residual_sum = std::numeric_limits<double>::max();
};

// Measure the support of a model by counting the number of inliers and
// summing all inlier residuals. The support is better if it has more inliers
// and a smaller residual sum.
struct InlierSupportMeasurer {
  // Compute the support of the residuals.
  Support Evaluate(const std::vector<double>& residuals,
                   const double max_residual);

  // Compare the two supports and return the better support.
  bool Compare(const Support& support1, const Support& support2);
};

// Measure the support of a model by its fitness to the data as used in MSAC.
// A support is better if it has a smaller MSAC score.
struct MEstimatorSupportMeasurer {
  // Compute the support of the residuals.
  Support Evaluate(const std::vector<double>& residuals,
                   const double max_residual);

  // Compare the two supports and return the better support.
  bool Compare(const Support& support1, const Support& support2);
};