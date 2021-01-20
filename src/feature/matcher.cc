#include "src/feature/matcher.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/eigen.hpp>

#include "FLANN/flann.hpp"
#include "src/util/logging.h"

const float TH_HIGH = 100;

float ORBDescriptorDistance(const Eigen::Matrix<uint8_t, 1, Eigen::Dynamic, Eigen::RowMajor> &a, const Eigen::Matrix<uint8_t, 1, Eigen::Dynamic, Eigen::RowMajor> &b) {
    float dist = 0.0;
    cv::Mat a_mat;
    cv::eigen2cv(a, a_mat);
    cv::Mat b_mat;
    cv::eigen2cv(b, b_mat);

    const int *pa = a_mat.ptr<int32_t>();
    const int *pb = b_mat.ptr<int32_t>();

    for(int i=0; i<8; i++, pa++, pb++)
    {
        unsigned  int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}

float SiftDescriptorDistance(const Eigen::Matrix<uint8_t, 1, Eigen::Dynamic, Eigen::RowMajor> &a, const Eigen::Matrix<uint8_t, 1, Eigen::Dynamic, Eigen::RowMajor> &b) {
    const Eigen::Matrix<int, 1, 128> a_int = a.cast<int>();
    const Eigen::Matrix<int, 1, 128> b_int = b.cast<int>();
    float dist = a_int.dot(b_int);
    const float kDistNorm = 1.0 / (512.0f * 512.0f);
    return std::acos(std::min(dist * kDistNorm, 1.0f));
}

Eigen::MatrixXf ComputeDistanceMatrix(const FeatureDescriptors& descriptors1,
                                      const FeatureDescriptors& descriptors2,
                                      const std::function<float(const Eigen::Matrix<uint8_t, 1, Eigen::Dynamic, Eigen::RowMajor> &, 
                                        const Eigen::Matrix<uint8_t, 1, Eigen::Dynamic, Eigen::RowMajor> &)>& func_distance) {
    CHECK_EQ(descriptors1.cols(), descriptors2.cols());
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dists(descriptors1.rows(), descriptors2.rows());
    for (FeatureDescriptors::Index i1 = 0; i1 < descriptors1.rows(); ++i1) {
        for (FeatureDescriptors::Index i2 = 0; i2 < descriptors2.rows(); ++i2) {
            dists(i1, i2) = func_distance(descriptors1.row(i1), descriptors2.row(i2));
        }
    }
    return dists;
}

FirstSecondFeatureMatches BFComputeFeatureMatches(const FeatureDescriptors& descriptors1,
                                                  const FeatureDescriptors& descriptors2,
                                                  const std::string& feature_type) {
    CHECK_EQ(descriptors1.cols(), descriptors2.cols());
    assert(feature_type == "sift" || feature_type == "orb");
    std::function<float(const Eigen::Matrix<uint8_t, 1, Eigen::Dynamic, Eigen::RowMajor> &, 
        const Eigen::Matrix<uint8_t, 1, Eigen::Dynamic, Eigen::RowMajor> &)> func_distance;
    if (feature_type == "sift")
        func_distance = SiftDescriptorDistance;
    if (feature_type == "orb")
        func_distance = ORBDescriptorDistance;
    Eigen::MatrixXf dists = ComputeDistanceMatrix(descriptors1, descriptors2, func_distance);
    FirstSecondFeatureMatches matches;

    for (Eigen::Index i1 = 0; i1 < dists.rows(); ++i1) {
        int best_i2 = -1;
        float best_dist = TH_HIGH;
        int second_best_i2 = -1;
        float second_best_dist = TH_HIGH;
        for (Eigen::Index i2 = 0; i2 < dists.cols(); ++i2) {
            const float dist = dists(i1, i2);
            if (dist < best_dist) {
                second_best_i2 = best_i2;
                second_best_dist = best_dist;
                best_i2 = i2;
                best_dist = dist;
            } else if (dist < second_best_dist) {
                second_best_dist = dist;
                second_best_i2 = i2;
            }
        }

        std::pair<FeatureMatch, FeatureMatch> match(FeatureMatch(i1, best_i2, best_dist), FeatureMatch(i1, second_best_i2, second_best_dist));
        matches.push_back(match);
    }
    return matches;
}

FirstSecondFeatureMatches FLANNComputeFeatureMatches(const FeatureDescriptors& descriptors1,
                                                     const FeatureDescriptors& descriptors2,
                                                     const std::string& feature_type) {
    assert(feature_type == "sift" || feature_type == "orb");
    std::function<float(const Eigen::Matrix<uint8_t, 1, Eigen::Dynamic, Eigen::RowMajor> &, 
        const Eigen::Matrix<uint8_t, 1, Eigen::Dynamic, Eigen::RowMajor> &)> func_distance;
    if (feature_type == "sift")
        func_distance = SiftDescriptorDistance;
    if (feature_type == "orb")
        func_distance = ORBDescriptorDistance;
    FirstSecondFeatureMatches matches;
    if (descriptors1.rows() == 0 || descriptors2.rows() == 0)
        return matches;
    CHECK_EQ(descriptors1.cols(), descriptors2.cols());
    
    const size_t kNumNearestNeighbors = 2;
    const size_t kNumTreesInForest = 4;

    const size_t num_nearest_neighbors =
        std::min(kNumNearestNeighbors, static_cast<size_t>(descriptors2.rows()));
    
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> indices;
    indices.resize(descriptors1.rows(), num_nearest_neighbors);
    const flann::Matrix<uint8_t> query_matrix(const_cast<uint8_t*>(descriptors1.data()),
                                                                    descriptors1.rows(), descriptors1.cols());
    const flann::Matrix<uint8_t> database_matrix(const_cast<uint8_t*>(descriptors2.data()),
                                                                        descriptors2.rows(), descriptors2.cols());

    flann::Matrix<int> indices_matrix(indices.data(), descriptors1.rows(), num_nearest_neighbors);
    if (feature_type == "sift") {
        std::vector<float> distances_vector(descriptors1.rows() * num_nearest_neighbors);
        flann::Matrix<float> distances_matrix(distances_vector.data(), descriptors1.rows(), num_nearest_neighbors);
        flann::Index<flann::L2<uint8_t> > index(
            database_matrix, flann::KDTreeIndexParams(kNumTreesInForest));
        index.buildIndex();
        index.knnSearch(query_matrix, indices_matrix, distances_matrix,
                        num_nearest_neighbors, flann::SearchParams(descriptors1.cols()));
    }
    if (feature_type == "orb") {
        std::vector<unsigned int> distances_vector(descriptors1.rows() * num_nearest_neighbors);
        flann::Matrix<unsigned int> distances_matrix(distances_vector.data(), descriptors1.rows(), num_nearest_neighbors);
        flann::Index<flann::Hamming<uint8_t> > index(
            database_matrix, flann::LshIndexParams(6, 12, 1));
        index.buildIndex();
        index.knnSearch(query_matrix, indices_matrix, distances_matrix,
                        num_nearest_neighbors, flann::SearchParams(descriptors1.cols()));
    }

    for (Eigen::Index query_index = 0; query_index < indices.rows(); ++query_index) {
        int best_i2 = -1;
        float best_dist = TH_HIGH;
        int second_best_i2 = -1;
        float second_best_dist = TH_HIGH;
        for (Eigen::Index k = 0; k < indices.cols(); ++k) {
            const Eigen::Index database_index = indices.coeff(query_index, k);
            float dist = func_distance(descriptors1.row(query_index), descriptors2.row(database_index));
            if (dist < best_dist) {
                second_best_i2 = best_i2;
                second_best_dist = best_dist;
                best_i2 = database_index;
                best_dist = dist;
            } else if (dist < second_best_dist) {
                second_best_dist = dist;
                second_best_i2 = database_index;
            }
        }

        std::pair<FeatureMatch, FeatureMatch> match(FeatureMatch(query_index, best_i2, best_dist), FeatureMatch(query_index, second_best_i2, second_best_dist));
        matches.push_back(match);
    }
    return matches;
}