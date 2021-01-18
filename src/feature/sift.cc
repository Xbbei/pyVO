#include "src/feature/sift.h"

#include <array>
#include <fstream>
#include <memory>

#include <GL/glew.h>

#include "src/feature/utils.h"
#include "src/util/math.h"
#include "src/util/misc.h"
#include "src/util/logging.h"

void WarnDarknessAdaptivityNotAvailable() {
  std::cout << "WARNING: Darkness adaptivity only available for GLSL SiftGPU."
            << std::endl;
}

// VLFeat uses a different convention to store its descriptors. This transforms
// the VLFeat format into the original SIFT format that is also used by SiftGPU.
FeatureDescriptors TransformVLFeatToUBCFeatureDescriptors(
    const FeatureDescriptors& vlfeat_descriptors) {
  FeatureDescriptors ubc_descriptors(vlfeat_descriptors.rows(),
                                     vlfeat_descriptors.cols());
  const std::array<int, 8> q{{0, 7, 6, 5, 4, 3, 2, 1}};
  for (FeatureDescriptors::Index n = 0; n < vlfeat_descriptors.rows(); ++n) {
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        for (int k = 0; k < 8; ++k) {
          ubc_descriptors(n, 8 * (j + 4 * i) + q[k]) =
              vlfeat_descriptors(n, 8 * (j + 4 * i) + k);
        }
      }
    }
  }
  return ubc_descriptors;
}

bool SiftExtractionOptions::Check() const {
  if (use_gpu) {
    CHECK_OPTION_GT(CSVToVector<int>(gpu_index).size(), 0);
  }
  CHECK_OPTION_GT(max_image_size, 0);
  CHECK_OPTION_GT(max_num_features, 0);
  CHECK_OPTION_GT(octave_resolution, 0);
  CHECK_OPTION_GT(peak_threshold, 0.0);
  CHECK_OPTION_GT(edge_threshold, 0.0);
  CHECK_OPTION_GT(max_num_orientations, 0);
  if (domain_size_pooling) {
    CHECK_OPTION_GT(dsp_min_scale, 0);
    CHECK_OPTION_GE(dsp_max_scale, dsp_min_scale);
    CHECK_OPTION_GT(dsp_num_scales, 0);
  }
  return true;
}

SiftExtract::SiftExtract() {
  Reset();
}

SiftExtract::SiftExtract(const SiftExtractionOptions& options) {
  Reset(options);
}

void SiftExtract::Reset(const SiftExtractionOptions& options) {
  keypoints_.clear();
  options_ = options;
}

bool SiftExtract::CreateSiftGPUExtractor() {
  CHECK(options_.Check());

  std::vector<int> gpu_indices = CSVToVector<int>(options_.gpu_index);
  CHECK_EQ(gpu_indices.size(), 1) << "SiftGPU can only run on one GPU";

  std::vector<std::string> sift_gpu_args;

  sift_gpu_args.push_back("./sift_gpu");

#ifdef CUDA_ENABLED
  // Use CUDA version by default if darkness adaptivity is disabled.
  if (!options_.darkness_adaptivity && gpu_indices[0] < 0) {
    gpu_indices[0] = 0;
  }

  if (gpu_indices[0] >= 0) {
    sift_gpu_args.push_back("-cuda");
    sift_gpu_args.push_back(std::to_string(gpu_indices[0]));
  }
#endif  // CUDA_ENABLED

  // Darkness adaptivity (hidden feature). Significantly improves
  // distribution of features. Only available in GLSL version.
  if (options_.darkness_adaptivity) {
    if (gpu_indices[0] >= 0) {
      WarnDarknessAdaptivityNotAvailable();
    }
    sift_gpu_args.push_back("-da");
  }

  // No verbose logging.
  sift_gpu_args.push_back("-v");
  sift_gpu_args.push_back("0");

  // Fixed maximum image dimension.
  sift_gpu_args.push_back("-maxd");
  sift_gpu_args.push_back(std::to_string(options_.max_image_size));

  // Keep the highest level features.
  sift_gpu_args.push_back("-tc2");
  sift_gpu_args.push_back(std::to_string(options_.max_num_features));

  // First octave level.
  sift_gpu_args.push_back("-fo");
  sift_gpu_args.push_back(std::to_string(options_.first_octave));

  // Number of octave levels.
  sift_gpu_args.push_back("-d");
  sift_gpu_args.push_back(std::to_string(options_.octave_resolution));

  // Peak threshold.
  sift_gpu_args.push_back("-t");
  sift_gpu_args.push_back(std::to_string(options_.peak_threshold));

  // Edge threshold.
  sift_gpu_args.push_back("-e");
  sift_gpu_args.push_back(std::to_string(options_.edge_threshold));

  if (options_.upright) {
    // Fix the orientation to 0 for upright features.
    sift_gpu_args.push_back("-ofix");
    // Maximum number of orientations.
    sift_gpu_args.push_back("-mo");
    sift_gpu_args.push_back("1");
  } else {
    // Maximum number of orientations.
    sift_gpu_args.push_back("-mo");
    sift_gpu_args.push_back(std::to_string(options_.max_num_orientations));
  }

  std::vector<const char*> sift_gpu_args_cstr;
  sift_gpu_args_cstr.reserve(sift_gpu_args.size());
  for (const auto& arg : sift_gpu_args) {
    sift_gpu_args_cstr.push_back(arg.c_str());
  }

  sift_gpu_.ParseParam(sift_gpu_args_cstr.size(), sift_gpu_args_cstr.data());

  sift_gpu_.gpu_index = gpu_indices[0];

  return sift_gpu_.VerifyContextGL() == SiftGPU::SIFTGPU_FULL_SUPPORTED;
}

bool SiftExtract::ExtractSiftFeaturesGPU(const cv::Mat& bitmap) {
  CHECK(options_.Check());
  CHECK(bitmap.type() == CV_8UC1);
  CHECK_EQ(options_.max_image_size, sift_gpu_.GetMaxDimension());

  CHECK(!options_.estimate_affine_shape);
  CHECK(!options_.domain_size_pooling);

  // Note, that this produces slightly different results than using SiftGPU
  // directly for RGB->GRAY conversion, since it uses different weights.
  const std::vector<uint8_t> bitmap_raw_bits = (std::vector<uint8_t>)bitmap.reshape(1, 1);
  const int code =
      sift_gpu_.RunSIFT(bitmap.rows, bitmap.cols,
                        bitmap_raw_bits.data(), GL_LUMINANCE, GL_UNSIGNED_BYTE);

  const int kSuccessCode = 1;
  if (code != kSuccessCode) {
    return false;
  }

  const size_t num_features = static_cast<size_t>(sift_gpu_.GetFeatureNum());

  std::vector<SiftKeypoint> keypoints_data(num_features);

  // Eigen's default is ColMajor, but SiftGPU stores result as RowMajor.
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      descriptors_float(num_features, 128);

  // Download the extracted keypoints and descriptors.
  sift_gpu_.GetFeatureVector(keypoints_data.data(), descriptors_float.data());

  keypoints_.resize(num_features);
  for (size_t i = 0; i < num_features; ++i) {
    keypoints_[i] = FeatureKeypoint(keypoints_data[i].x, keypoints_data[i].y,
                                      keypoints_data[i].s, keypoints_data[i].o);
  }

  // Save and normalize the descriptors.
  if (options_.normalization == SiftExtractionOptions::Normalization::L2) {
    descriptors_float = L2NormalizeFeatureDescriptors(descriptors_float);
  } else if (options_.normalization ==
             SiftExtractionOptions::Normalization::L1_ROOT) {
    descriptors_float = L1RootNormalizeFeatureDescriptors(descriptors_float);
  } else {
    LOG(FATAL) << "Normalization type not supported";
  }

  descriptors_ = FeatureDescriptorsToUnsignedByte(descriptors_float);

  return true;
}

void SiftExtract::LoadSiftFeaturesFromTextFile(const std::string& path) {
  std::ifstream file(path.c_str());
  CHECK(file.is_open()) << path;

  std::string line;
  std::string item;

  std::getline(file, line);
  std::stringstream header_line_stream(line);

  std::getline(header_line_stream >> std::ws, item, ' ');
  const uint32_t num_features = std::stoul(item);

  std::getline(header_line_stream >> std::ws, item, ' ');
  const size_t dim = std::stoul(item);

  CHECK_EQ(dim, 128) << "SIFT features must have 128 dimensions";

  keypoints_.resize(num_features);
  descriptors_.resize(num_features, dim);

  for (size_t i = 0; i < num_features; ++i) {
    std::getline(file, line);
    std::stringstream feature_line_stream(line);

    std::getline(feature_line_stream >> std::ws, item, ' ');
    const float x = std::stold(item);

    std::getline(feature_line_stream >> std::ws, item, ' ');
    const float y = std::stold(item);

    std::getline(feature_line_stream >> std::ws, item, ' ');
    const float scale = std::stold(item);

    std::getline(feature_line_stream >> std::ws, item, ' ');
    const float orientation = std::stold(item);

    keypoints_[i] = FeatureKeypoint(x, y, scale, orientation);

    // Descriptor
    for (size_t j = 0; j < dim; ++j) {
      std::getline(feature_line_stream >> std::ws, item, ' ');
      const float value = std::stod(item);
      CHECK_GE(value, 0);
      CHECK_LE(value, 255);
      descriptors_(i, j) = TruncateCast<float, uint8_t>(value);
    }
  }
}

bool SiftExtract::ExtractSiftFeaturesCPU(const cv::Mat& bitmap) {
  CHECK(options_.Check());
  CHECK(bitmap.type() == CV_8UC1);

  CHECK(!options_.estimate_affine_shape);
  CHECK(!options_.domain_size_pooling);

  if (options_.darkness_adaptivity) {
    WarnDarknessAdaptivityNotAvailable();
  }

  // Setup SIFT extractor.
  std::unique_ptr<VlSiftFilt, void (*)(VlSiftFilt*)> sift(
      vl_sift_new(bitmap.cols, bitmap.rows, options_.num_octaves,
                  options_.octave_resolution, options_.first_octave),
      &vl_sift_delete);
  if (!sift) {
    return false;
  }

  vl_sift_set_peak_thresh(sift.get(), options_.peak_threshold);
  vl_sift_set_edge_thresh(sift.get(), options_.edge_threshold);

  // Iterate through octaves.
  std::vector<size_t> level_num_features;
  std::vector<FeatureKeypoints> level_keypoints;
  std::vector<FeatureDescriptors> level_descriptors;
  bool first_octave = true;
  while (true) {
    if (first_octave) {
      const std::vector<uint8_t> data_uint8 = (std::vector<uint8_t>)bitmap.reshape(1, 1);
      std::vector<float> data_float(data_uint8.size());
      for (size_t i = 0; i < data_uint8.size(); ++i) {
        data_float[i] = static_cast<float>(data_uint8[i]) / 255.0f;
      }
      if (vl_sift_process_first_octave(sift.get(), data_float.data())) {
        break;
      }
      first_octave = false;
    } else {
      if (vl_sift_process_next_octave(sift.get())) {
        break;
      }
    }

    // Detect keypoints.
    vl_sift_detect(sift.get());

    // Extract detected keypoints.
    const VlSiftKeypoint* vl_keypoints = vl_sift_get_keypoints(sift.get());
    const int num_keypoints = vl_sift_get_nkeypoints(sift.get());
    if (num_keypoints == 0) {
      continue;
    }

    // Extract features with different orientations per DOG level.
    size_t level_idx = 0;
    int prev_level = -1;
    for (int i = 0; i < num_keypoints; ++i) {
      if (vl_keypoints[i].is != prev_level) {
        if (i > 0) {
          // Resize containers of previous DOG level.
          level_keypoints.back().resize(level_idx);
          level_descriptors.back().conservativeResize(level_idx, 128);
        }

        // Add containers for new DOG level.
        level_idx = 0;
        level_num_features.push_back(0);
        level_keypoints.emplace_back(options_.max_num_orientations *
                                     num_keypoints);
        level_descriptors.emplace_back(
            options_.max_num_orientations * num_keypoints, 128);
      }

      level_num_features.back() += 1;
      prev_level = vl_keypoints[i].is;

      // Extract feature orientations.
      double angles[4];
      int num_orientations;
      if (options_.upright) {
        num_orientations = 1;
        angles[0] = 0.0;
      } else {
        num_orientations = vl_sift_calc_keypoint_orientations(
            sift.get(), angles, &vl_keypoints[i]);
      }

      // Note that this is different from SiftGPU, which selects the top
      // global maxima as orientations while this selects the first two
      // local maxima. It is not clear which procedure is better.
      const int num_used_orientations =
          std::min(num_orientations, options_.max_num_orientations);

      for (int o = 0; o < num_used_orientations; ++o) {
        level_keypoints.back()[level_idx] =
            FeatureKeypoint(vl_keypoints[i].x + 0.5f, vl_keypoints[i].y + 0.5f,
                            vl_keypoints[i].sigma, angles[o]);
        {
          Eigen::MatrixXf desc(1, 128);
          vl_sift_calc_keypoint_descriptor(sift.get(), desc.data(),
                                           &vl_keypoints[i], angles[o]);
          if (options_.normalization ==
              SiftExtractionOptions::Normalization::L2) {
            desc = L2NormalizeFeatureDescriptors(desc);
          } else if (options_.normalization ==
                     SiftExtractionOptions::Normalization::L1_ROOT) {
            desc = L1RootNormalizeFeatureDescriptors(desc);
          } else {
            LOG(FATAL) << "Normalization type not supported";
          }

          level_descriptors.back().row(level_idx) =
              FeatureDescriptorsToUnsignedByte(desc);
        }

        level_idx += 1;
      }
    }

    // Resize containers for last DOG level in octave.
    level_keypoints.back().resize(level_idx);
    {
      level_descriptors.back().conservativeResize(level_idx, 128);
    }
  }

  // Determine how many DOG levels to keep to satisfy max_num_features option.
  int first_level_to_keep = 0;
  int num_features = 0;
  int num_features_with_orientations = 0;
  for (int i = level_keypoints.size() - 1; i >= 0; --i) {
    num_features += level_num_features[i];
    num_features_with_orientations += level_keypoints[i].size();
    if (num_features > options_.max_num_features) {
      first_level_to_keep = i;
      break;
    }
  }

  // Extract the features to be kept.
  {
    size_t k = 0;
    keypoints_.resize(num_features_with_orientations);
    for (size_t i = first_level_to_keep; i < level_keypoints.size(); ++i) {
      for (size_t j = 0; j < level_keypoints[i].size(); ++j) {
        keypoints_[k] = level_keypoints[i][j];
        k += 1;
      }
    }
  }

  // Compute the descriptors for the detected keypoints.
  {
    size_t k = 0;
    descriptors_.resize(num_features_with_orientations, 128);
    for (size_t i = first_level_to_keep; i < level_keypoints.size(); ++i) {
      for (size_t j = 0; j < level_keypoints[i].size(); ++j) {
        descriptors_.row(k) = level_descriptors[i].row(j);
        k += 1;
      }
    }
    descriptors_ = TransformVLFeatToUBCFeatureDescriptors(descriptors_);
  }

  return true;
}

bool SiftExtract::ExtractCovariantSiftFeaturesCPU(const cv::Mat& bitmap) {
  CHECK(options_.Check());
  CHECK(bitmap.type() == CV_8UC1);

  if (options_.darkness_adaptivity) {
    WarnDarknessAdaptivityNotAvailable();
  }

  // Setup covariant SIFT detector.
  std::unique_ptr<VlCovDet, void (*)(VlCovDet*)> covdet(
      vl_covdet_new(VL_COVDET_METHOD_DOG), &vl_covdet_delete);
  if (!covdet) {
    return false;
  }

  const int kMaxOctaveResolution = 1000;
  CHECK_LE(options_.octave_resolution, kMaxOctaveResolution);

  vl_covdet_set_first_octave(covdet.get(), options_.first_octave);
  vl_covdet_set_octave_resolution(covdet.get(), options_.octave_resolution);
  vl_covdet_set_peak_threshold(covdet.get(), options_.peak_threshold);
  vl_covdet_set_edge_threshold(covdet.get(), options_.edge_threshold);

  {
    const std::vector<uint8_t> data_uint8 = (std::vector<uint8_t>)bitmap.reshape(1, 1);
    std::vector<float> data_float(data_uint8.size());
    for (size_t i = 0; i < data_uint8.size(); ++i) {
      data_float[i] = static_cast<float>(data_uint8[i]) / 255.0f;
    }
    vl_covdet_put_image(covdet.get(), data_float.data(), bitmap.cols,
                        bitmap.rows);
  }

  vl_covdet_detect(covdet.get(), options_.max_num_features);

  if (!options_.upright) {
    if (options_.estimate_affine_shape) {
      vl_covdet_extract_affine_shape(covdet.get());
    } else {
      vl_covdet_extract_orientations(covdet.get());
    }
  }

  const int num_features = vl_covdet_get_num_features(covdet.get());
  VlCovDetFeature* features = vl_covdet_get_features(covdet.get());

  // Sort features according to detected octave and scale.
  std::sort(
      features, features + num_features,
      [](const VlCovDetFeature& feature1, const VlCovDetFeature& feature2) {
        if (feature1.o == feature2.o) {
          return feature1.s > feature2.s;
        } else {
          return feature1.o > feature2.o;
        }
      });

  const size_t max_num_features = static_cast<size_t>(options_.max_num_features);

  // Copy detected keypoints and clamp when maximum number of features reached.
  int prev_octave_scale_idx = std::numeric_limits<int>::max();
  for (int i = 0; i < num_features; ++i) {
    FeatureKeypoint keypoint;
    keypoint.x = features[i].frame.x + 0.5;
    keypoint.y = features[i].frame.y + 0.5;
    keypoint.a11 = features[i].frame.a11;
    keypoint.a12 = features[i].frame.a12;
    keypoint.a21 = features[i].frame.a21;
    keypoint.a22 = features[i].frame.a22;
    keypoints_.push_back(keypoint);

    const int octave_scale_idx =
        features[i].o * kMaxOctaveResolution + features[i].s;
    CHECK_LE(octave_scale_idx, prev_octave_scale_idx);

    if (octave_scale_idx != prev_octave_scale_idx &&
        keypoints_.size() >= max_num_features) {
      break;
    }

    prev_octave_scale_idx = octave_scale_idx;
  }

  // Compute the descriptors for the detected keypoints.
  {
    descriptors_.resize(keypoints_.size(), 128);

    const size_t kPatchResolution = 15;
    const size_t kPatchSide = 2 * kPatchResolution + 1;
    const double kPatchRelativeExtent = 7.5;
    const double kPatchRelativeSmoothing = 1;
    const double kPatchStep = kPatchRelativeExtent / kPatchResolution;
    const double kSigma =
        kPatchRelativeExtent / (3.0 * (4 + 1) / 2) / kPatchStep;

    std::vector<float> patch(kPatchSide * kPatchSide);
    std::vector<float> patchXY(2 * kPatchSide * kPatchSide);

    float dsp_min_scale = 1;
    float dsp_scale_step = 0;
    int dsp_num_scales = 1;
    if (options_.domain_size_pooling) {
      dsp_min_scale = options_.dsp_min_scale;
      dsp_scale_step = (options_.dsp_max_scale - options_.dsp_min_scale) /
                       options_.dsp_num_scales;
      dsp_num_scales = options_.dsp_num_scales;
    }

    Eigen::Matrix<float, Eigen::Dynamic, 128, Eigen::RowMajor>
        scaled_descriptors(dsp_num_scales, 128);

    std::unique_ptr<VlSiftFilt, void (*)(VlSiftFilt*)> sift(
        vl_sift_new(16, 16, 1, 3, 0), &vl_sift_delete);
    if (!sift) {
      return false;
    }

    vl_sift_set_magnif(sift.get(), 3.0);

    for (size_t i = 0; i < keypoints_.size(); ++i) {
      for (int s = 0; s < dsp_num_scales; ++s) {
        const double dsp_scale = dsp_min_scale + s * dsp_scale_step;

        VlFrameOrientedEllipse scaled_frame = features[i].frame;
        scaled_frame.a11 *= dsp_scale;
        scaled_frame.a12 *= dsp_scale;
        scaled_frame.a21 *= dsp_scale;
        scaled_frame.a22 *= dsp_scale;

        vl_covdet_extract_patch_for_frame(
            covdet.get(), patch.data(), kPatchResolution, kPatchRelativeExtent,
            kPatchRelativeSmoothing, scaled_frame);

        vl_imgradient_polar_f(patchXY.data(), patchXY.data() + 1, 2,
                              2 * kPatchSide, patch.data(), kPatchSide,
                              kPatchSide, kPatchSide);

        vl_sift_calc_raw_descriptor(sift.get(), patchXY.data(),
                                    scaled_descriptors.row(s).data(),
                                    kPatchSide, kPatchSide, kPatchResolution,
                                    kPatchResolution, kSigma, 0);
      }

      Eigen::Matrix<float, 1, 128> descriptor;
      if (options_.domain_size_pooling) {
        descriptor = scaled_descriptors.colwise().mean();
      } else {
        descriptor = scaled_descriptors;
      }

      if (options_.normalization == SiftExtractionOptions::Normalization::L2) {
        descriptor = L2NormalizeFeatureDescriptors(descriptor);
      } else if (options_.normalization ==
                 SiftExtractionOptions::Normalization::L1_ROOT) {
        descriptor = L1RootNormalizeFeatureDescriptors(descriptor);
      } else {
        LOG(FATAL) << "Normalization type not supported";
      }

      descriptors_.row(i) = FeatureDescriptorsToUnsignedByte(descriptor);
    }

    descriptors_ = TransformVLFeatToUBCFeatureDescriptors(descriptors_);
  }

  return true;
}

bool SiftExtract::ExtractSiftFeatures(const cv::Mat& bitmap) {
  keypoints_.clear();
  if (options_.use_gpu) {
    if (!CreateSiftGPUExtractor()) {
      std::cout << "ERROR: SiftGPU not fully supported." << std::endl;
      return false;
    }
  }
  bool success = false;
  if (options_.estimate_affine_shape ||
      options_.domain_size_pooling) {
    success = ExtractCovariantSiftFeaturesCPU(bitmap);
  } else if (options_.use_gpu) {
    success = ExtractSiftFeaturesGPU(bitmap);
  } else {
    success = ExtractSiftFeaturesCPU(bitmap);
  }
  return success;
}

FeatureKeypoints SiftExtract::getKeyPoints() const {
  return keypoints_;
}

FeatureDescriptors SiftExtract::getDescriptors() const {
  return descriptors_;
}