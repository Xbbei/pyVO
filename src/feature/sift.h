#pragma once

#include <opencv2/opencv.hpp>
#include "src/feature/types.h"

#include "SiftGPU/SiftGPU.h"
#include "VLFeat/covdet.h"
#include "VLFeat/sift.h"

struct SiftExtractionOptions {
  // Number of threads for feature extraction.
  int num_threads = -1;

  // Whether to use the GPU for feature extraction.
  bool use_gpu = true;

  // Index of the GPU used for feature extraction. For multi-GPU extraction,
  // you should separate multiple GPU indices by comma, e.g., "0,1,2,3".
  std::string gpu_index = "-1";

  // Maximum image size, otherwise image will be down-scaled.
  int max_image_size = 3200;

  // Maximum number of features to detect, keeping larger-scale features.
  int max_num_features = 8192;

  // First octave in the pyramid, i.e. -1 upsamples the image by one level.
  int first_octave = -1;

  // Number of octaves.
  int num_octaves = 4;

  // Number of levels per octave.
  int octave_resolution = 3;

  // Peak threshold for detection.
  double peak_threshold = 0.02 / octave_resolution;

  // Edge threshold for detection.
  double edge_threshold = 10.0;

  // Estimate affine shape of SIFT features in the form of oriented ellipses as
  // opposed to original SIFT which estimates oriented disks.
  bool estimate_affine_shape = false;

  // Maximum number of orientations per keypoint if not estimate_affine_shape.
  int max_num_orientations = 2;

  // Fix the orientation to 0 for upright features.
  bool upright = false;

  // Whether to adapt the feature detection depending on the image darkness.
  // Note that this feature is only available in the OpenGL SiftGPU version.
  bool darkness_adaptivity = false;

  // Domain-size pooling parameters. Domain-size pooling computes an average
  // SIFT descriptor across multiple scales around the detected scale. This was
  // proposed in "Domain-Size Pooling in Local Descriptors and Network
  // Architectures", J. Dong and S. Soatto, CVPR 2015. This has been shown to
  // outperform other SIFT variants and learned descriptors in "Comparative
  // Evaluation of Hand-Crafted and Learned Local Features", Schönberger,
  // Hardmeier, Sattler, Pollefeys, CVPR 2016.
  bool domain_size_pooling = false;
  double dsp_min_scale = 1.0 / 6.0;
  double dsp_max_scale = 3.0;
  int dsp_num_scales = 10;

  enum class Normalization {
    // L1-normalizes each descriptor followed by element-wise square rooting.
    // This normalization is usually better than standard L2-normalization.
    // See "Three things everyone should know to improve object retrieval",
    // Relja Arandjelovic and Andrew Zisserman, CVPR 2012.
    L1_ROOT,
    // Each vector is L2-normalized.
    L2,
  };
  Normalization normalization = Normalization::L1_ROOT;

  bool Check() const;
};

class SiftExtract {
public:
  SiftExtract();
  SiftExtract(const SiftExtractionOptions& options);
  void Reset(const SiftExtractionOptions& options = SiftExtractionOptions());

  FeatureKeypoints getKeyPoints() const;
  FeatureDescriptors getDescriptors() const;

  bool ExtractSiftFeatures(const cv::Mat& bitmap);
private:
  FeatureKeypoints keypoints_;
  FeatureDescriptors descriptors_;
  SiftExtractionOptions options_;
  SiftGPU sift_gpu_;

  // Create a SiftGPU feature extractor. The same SiftGPU instance can be used to
  // extract features for multiple images. Note a OpenGL context must be made
  // current in the thread of the caller. If the gpu_index is not -1, the CUDA
  // version of SiftGPU is used, which produces slightly different results
  // than the OpenGL implementation.
  bool CreateSiftGPUExtractor();

  // Extract SIFT features for the given image on the GPU.
  // SiftGPU must already be initialized using `CreateSiftGPU`.
  bool ExtractSiftFeaturesGPU(const cv::Mat& bitmap);

  // Load keypoints and descriptors from text file in the following format:
  //
  //    LINE_0:            NUM_FEATURES DIM
  //    LINE_1:            X Y SCALE ORIENTATION D_1 D_2 D_3 ... D_DIM
  //    LINE_I:            ...
  //    LINE_NUM_FEATURES: X Y SCALE ORIENTATION D_1 D_2 D_3 ... D_DIM
  //
  // where the first line specifies the number of features and the descriptor
  // dimensionality followed by one line per feature: X, Y, SCALE, ORIENTATION are
  // of type float and D_J represent the descriptor in the range [0, 255].
  //
  // For example:
  //
  //    2 4
  //    0.32 0.12 1.23 1.0 1 2 3 4
  //    0.32 0.12 1.23 1.0 1 2 3 4
  //
  void LoadSiftFeaturesFromTextFile(const std::string& path);

  // Extract SIFT features for the given image on the CPU. Only extract
  // descriptors if the given input is not NULL.
  bool ExtractSiftFeaturesCPU(const cv::Mat& bitmap);
  bool ExtractCovariantSiftFeaturesCPU(const cv::Mat& bitmap);
};