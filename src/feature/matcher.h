#pragma once

#include <functional>

#include "src/feature/types.h"

// Bit set count operation from
// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
float ORBDescriptorDistance(const Eigen::Matrix<uint8_t, 1, Eigen::Dynamic, Eigen::RowMajor> &a, const Eigen::Matrix<uint8_t, 1, Eigen::Dynamic, Eigen::RowMajor> &b);

float SiftDescriptorDistance(const Eigen::Matrix<uint8_t, 1, Eigen::Dynamic, Eigen::RowMajor> &a, const Eigen::Matrix<uint8_t, 1, Eigen::Dynamic, Eigen::RowMajor> &b);

Eigen::MatrixXf ComputeDistanceMatrix(const FeatureDescriptors& descriptors1,
                                      const FeatureDescriptors& descriptors2,
                                      const std::function<float(const Eigen::Matrix<uint8_t, 1, Eigen::Dynamic, Eigen::RowMajor> &, 
                                        const Eigen::Matrix<uint8_t, 1, Eigen::Dynamic, Eigen::RowMajor> &)>& func_distance);

FirstSecondFeatureMatches BFComputeFeatureMatches(const FeatureDescriptors& descriptors1,
                                                  const FeatureDescriptors& descriptors2,
                                                  const std::string& feature_type);

FirstSecondFeatureMatches FLANNComputeFeatureMatches(const FeatureDescriptors& descriptors1,
                                                     const FeatureDescriptors& descriptors2,
                                                     const std::string& feature_type);