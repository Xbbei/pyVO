#pragma once

#include <vector>
#include <list>
#include <opencv/cv.h>

#include "src/feature/types.h"

class ExtractorNode
{
public:
    ExtractorNode():bNoMore(false){}

    void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4);

    std::vector<cv::KeyPoint> vKeys;
    cv::Point2i UL, UR, BL, BR;
    std::list<ExtractorNode>::iterator lit;
    bool bNoMore;
};

struct ORBExtractionOptions {
    // Number of features per image
    int nFeatures = 8192;
    // Scale factor between levels in the scale pyramid
    float scaleFactor = 1.2;
    // Number of levels in the scale pyramid
    int nLevels = 8;
    // Fast threshold
    // Image is divided in a grid. At each cell FAST are extracted imposing a minimum response.
    // Firstly we impose iniThFAST. If no corners are detected we impose a lower value minThFAST
    // You can lower these values if your images have low contrast
    int iniThFAST = 20;
    int minThFAST = 7;

    bool Check() const;
};

class ORBExtract {
public:
    enum {HARRIS_SCORE=0, FAST_SCORE=1 };

    ORBExtract();
    ORBExtract(const ORBExtractionOptions& options);
    void Reset(const ORBExtractionOptions& options = ORBExtractionOptions());

    FeatureKeypoints getKeyPoints() const;
    FeatureDescriptors getDescriptors() const;

    bool ExtractORBFeatures(const cv::Mat& bitmap);

    std::vector<cv::Mat> mvImagePyramid;

private:
    ORBExtractionOptions options_;

    std::vector<cv::Point> pattern;

    std::vector<int> mnFeaturesPerLevel;

    std::vector<int> umax;

    std::vector<float> mvScaleFactor;
    std::vector<float> mvInvScaleFactor;    
    std::vector<float> mvLevelSigma2;
    std::vector<float> mvInvLevelSigma2;

    FeatureKeypoints keypoints_;
    FeatureDescriptors descriptors_;

    std::vector<cv::KeyPoint> DistributeOctTree(const std::vector<cv::KeyPoint>& vToDistributeKeys, const int &minX,
        const int &maxX, const int &minY, const int &maxY, const int &N, const int &level);

    void ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint> >& allkeypoints);

    void ComputePyramid(cv::Mat image);
};