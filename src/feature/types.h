#pragma once

#include <vector>

#include <Eigen/Core>

#include "src/util/types.h"


const uint32_t kInvalidPoint2DIdx = std::numeric_limits<uint32_t>::max();

struct FeatureKeypoint {
  FeatureKeypoint();
  FeatureKeypoint(const float x, const float y);
  FeatureKeypoint(const float x, const float y, const float scale,
                  const float orientation);
  FeatureKeypoint(const float x, const float y, const float a11,
                  const float a12, const float a21, const float a22);

  static FeatureKeypoint FromParameters(const float x, const float y,
                                        const float scale_x,
                                        const float scale_y,
                                        const float orientation,
                                        const float shear);

  // Rescale the feature location and shape size by the given scale factor.
  void Rescale(const float scale);
  void Rescale(const float scale_x, const float scale_y);

  // Compute similarity shape parameters from affine shape.
  float ComputeScale() const;
  float ComputeScaleX() const;
  float ComputeScaleY() const;
  float ComputeOrientation() const;
  float ComputeShear() const;

  // Location of the feature, with the origin at the upper left image corner,
  // i.e. the upper left pixel has the coordinate (0.5, 0.5).
  float x;
  float y;

  // Affine shape of the feature.
  float a11;
  float a12;
  float a21;
  float a22;
};

typedef Eigen::Matrix<uint8_t, 1, Eigen::Dynamic, Eigen::RowMajor>
    FeatureDescriptor;

struct FeatureMatch {
  FeatureMatch()
      : point2D_idx1(kInvalidPoint2DIdx), point2D_idx2(kInvalidPoint2DIdx), distance(-1.0) {}
  FeatureMatch(const int point2D_idx1, const int point2D_idx2, const float distance)
      : point2D_idx1(point2D_idx1), point2D_idx2(point2D_idx2), distance(distance) {}

  // Feature index in first image.
  int point2D_idx1 = kInvalidPoint2DIdx;
  
  // Feature index in second image.
  int point2D_idx2 = kInvalidPoint2DIdx;

  // Feature distance between first and second images
  float distance;
};

typedef std::vector<FeatureKeypoint> FeatureKeypoints;
typedef Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    FeatureDescriptors;

typedef std::vector<FeatureMatch> FeatureMatches;
typedef std::vector<std::pair<FeatureMatch, FeatureMatch> > FirstSecondFeatureMatches;