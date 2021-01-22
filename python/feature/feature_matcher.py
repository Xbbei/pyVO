import sys
sys.path.append("../../build/")
import pyVO
import numpy as np
import cv2
import math
sys.path.append("../optim/")
from ransac import RansacPY
from loransac import LoRansacPY

# base class
class FeatureMatcher(object):
    def __init__(self, cross_check=False, ratio_test=0.8):
        super(FeatureMatcher, self).__init__()

        self.cross_check = cross_check
        self.ratio_test = ratio_test

    def match(self, kps1, des1, kps2, des2, ratio_test=None):
        pass

    # input: kps1 -> keypoints1, kps2 -> keypoints2
    # input: estimator -> essential_matrix estimator, fundamental_matrix estimator or homography_matrix estimator etl.
    def estimator_none_optim(self, kps1, kps2, estimator, confidence=0.9,
         max_residual=2.0*2.0, supportmeasurer=pyVO.optim.InlierSupportMeasurer):
        """Estimate the Matrix(H, E or F) According the correspondence points
            Returns:
                success, num_trials, support, inlier_mask, model
        """
        assert len(kps1) == len(kps2)
        num_samples = len(kps1)
        success = False
        support = pyVO.optim.Support()
        inlier_mask = np.zeros(len(kps1))
        model = None

        estimator_imp = estimator()
        supportmeasurer_imp = supportmeasurer()

        models = estimator_imp.Estimate(kps1, kps2)

        for sample_model in models:
            residuals = estimator_imp.Residuals(kps1, kps2, sample_model)
            assert len(residuals) == num_samples
            support_tmp = supportmeasurer_imp.Evaluate(residuals, max_residual)
            if supportmeasurer_imp.Compare(support_tmp, support):
                support = support_tmp
                model = sample_model
        
        num_inliers = 0
        residuals = estimator_imp.Residuals(kps1, kps2, model)
        assert len(residuals) == num_samples
        for i in range(num_samples):
            if residuals[i] < max_residual:
                num_inliers += 1
                inlier_mask[i] = 1
        success = num_inliers / num_samples > confidence
        return success, 1, support, inlier_mask, model

    def estimator_ransac(self, kps1, kps2, estimator, confidence=0.9,
         max_residual=2.0*2.0, supportmeasurer=pyVO.optim.InlierSupportMeasurer):
        """Estimate the Matrix(H, E or F) According RANSAC
            Returns:
                success, num_trials, support, inlier_mask, model
        """
        max_error = math.sqrt(max_residual)
        ransac = RansacPY(estimator, supportmeasurer, pyVO.optim.RandomSampler, max_error, confidence=confidence)
        return ransac.estimate(kps1, kps2)

    def estimator_loransac(self, kps1, kps2, estimator, loestimator, confidence=0.9,
         max_residual=2.0*2.0, supportmeasurer=pyVO.optim.InlierSupportMeasurer):
        """Estimate the Matrix(H, E or F) According LORANSAC
            Returns:
                success, num_trials, support, inlier_mask, model
        """
        max_error = math.sqrt(max_residual)
        loransac = LoRansacPY(estimator, loestimator, supportmeasurer, pyVO.optim.RandomSampler, max_error, confidence=confidence)
        return loransac.estimate(kps1, kps2)

    def matchbase(self, kps1, des1, kps2, des2, ratio_test, basematcher, feature_type):
        """Base Matcher for orb/sift bf/flann match
            Retruns:
                bfkps1, bfkps2 ------> match the keypoints and assert len(bfkps1) == len(bfkps2)
        """
        matches12 = basematcher(des1, des2, feature_type)
        if ratio_test is None:
            ratio_test = self.ratio_test
        if self.cross_check:
            matches21 = basematcher(des2, des1, feature_type)
        bfkps1 = []
        bfkps2 = []
        for match in matches12:
            if match[0].distance / match[1].distance > ratio_test:
                continue
            first = match[0].point2D_idx1
            second = match[0].point2D_idx2
            if self.cross_check and first != matches21[second][0].point2D_idx2:
                continue
            bfkps1.append(kps1[first])
            bfkps2.append(kps2[second])
        
        bfkps1 = np.array(bfkps1)
        bfkps2 = np.array(bfkps2)
        return bfkps1, bfkps2

class ORBFeatureMatcher(FeatureMatcher):
    def __init__(self, cross_check=False, ratio_test=0.8, match_type="flann"):
        super(ORBFeatureMatcher, self).__init__(cross_check=cross_check, ratio_test=ratio_test)

        assert match_type in ["flann", "bf"]
        self.match_type = match_type

    def match(self, kps1, des1, kps2, des2, ratio_test=None):
        """ORB Feature Match the keypoints according the distance between orb's descriptors, by bf/flann
            Retruns:
                bfkps1, bfkps2 ------> match the keypoints and assert len(bfkps1) == len(bfkps2)
        """
        if self.match_type == "bf":
            basematcher = pyVO.feature.BFComputeFeatureMatches
        if self.match_type == "flann":
            basematcher = pyVO.feature.FLANNComputeFeatureMatches
        return self.matchbase(kps1, des1, kps2, des2, ratio_test, basematcher, "orb")

class SiftFeatureMatcher(FeatureMatcher):
    def __init__(self, cross_check=False, ratio_test=0.8, match_type="flann"):
        super(SiftFeatureMatcher, self).__init__(cross_check=cross_check, ratio_test=ratio_test)

        assert match_type in ["flann", "bf"]
        self.match_type = match_type

    def match(self, kps1, des1, kps2, des2, ratio_test=None):
        """SIFT Feature Match the keypoints according the distance between sift's descriptors, by bf/flann
            Retruns:
                bfkps1, bfkps2 ------> match the keypoints and assert len(bfkps1) == len(bfkps2)
        """
        if self.match_type == "bf":
            basematcher = pyVO.feature.BFComputeFeatureMatches
        if self.match_type == "flann":
            basematcher = pyVO.feature.FLANNComputeFeatureMatches
        return self.matchbase(kps1, des1, kps2, des2, ratio_test, basematcher, "sift")  

if __name__ == "__main__":
    from orb import ORBExtractor
    from sift import SiftExtractor
    # ORB Match Test
    orb_extractor = ORBExtractor()
    image1 = cv2.imread("data/test1.jpg", -1)
    kps1, descriptors1 = orb_extractor.extract(image1)
    image2 = cv2.imread("data/test2.jpg", -1)
    kps2, descriptors2 = orb_extractor.extract(image2)
    print("orb extract done")

    # orb bf match
    bforbfeature_matcher = ORBFeatureMatcher(True, match_type="bf")
    matchkps1, matchkps2 = bforbfeature_matcher.match(kps1, descriptors1, kps2, descriptors2)
    image = np.concatenate([image1, image2], axis=1)
    for kp1, kp2 in zip(matchkps1, matchkps2):
        cv2.circle(image, (int(kp1[0]+0.5), int(kp1[1]+0.5)), 2, (0, 0, 255), 1)
        cv2.circle(image, (int(kp2[0]+0.5+image1.shape[1]), int(kp2[1]+0.5)), 2, (0, 255, 0), 1)
        cv2.line(image, (int(kp1[0]+0.5), int(kp1[1]+0.5)), (int(kp2[0]+0.5+image1.shape[1]), int(kp2[1]+0.5)), (255, 255, 0))
    cv2.imwrite("test/bforbmatch.jpg", image)
    print("orb bf match done")
    # orb flann match
    flannorbfeature_matcher = ORBFeatureMatcher(True, match_type="flann")
    matchkps1, matchkps2 = flannorbfeature_matcher.match(kps1, descriptors1, kps2, descriptors2)
    image = np.concatenate([image1, image2], axis=1)
    for kp1, kp2 in zip(matchkps1, matchkps2):
        cv2.circle(image, (int(kp1[0]+0.5), int(kp1[1]+0.5)), 2, (0, 0, 255), 1)
        cv2.circle(image, (int(kp2[0]+0.5+image1.shape[1]), int(kp2[1]+0.5)), 2, (0, 255, 0), 1)
        cv2.line(image, (int(kp1[0]+0.5), int(kp1[1]+0.5)), (int(kp2[0]+0.5+image1.shape[1]), int(kp2[1]+0.5)), (255, 255, 0))
    cv2.imwrite("test/flannorbmatch.jpg", image)
    print("orb flann match done")


    # Sift Match Test
    sift_extractor = SiftExtractor()
    kps1, descriptors1 = sift_extractor.extract(image1)
    kps2, descriptors2 = sift_extractor.extract(image2)
    print("sift extract done")

    # sift bf match
    bfsiftfeature_matcher = SiftFeatureMatcher(True, match_type="bf")
    matchkps1, matchkps2 = bfsiftfeature_matcher.match(kps1, descriptors1, kps2, descriptors2)
    image = np.concatenate([image1, image2], axis=1)
    for kp1, kp2 in zip(matchkps1, matchkps2):
        cv2.circle(image, (int(kp1[0]+0.5), int(kp1[1]+0.5)), 2, (0, 0, 255), 1)
        cv2.circle(image, (int(kp2[0]+0.5+image1.shape[1]), int(kp2[1]+0.5)), 2, (0, 255, 0), 1)
        cv2.line(image, (int(kp1[0]+0.5), int(kp1[1]+0.5)), (int(kp2[0]+0.5+image1.shape[1]), int(kp2[1]+0.5)), (255, 255, 0))
    cv2.imwrite("test/bfsiftmatch.jpg", image)
    print("sift bf match done")
    # sift flann match
    flannsiftfeature_matcher = SiftFeatureMatcher(True, match_type="flann")
    matchkps1, matchkps2 = flannsiftfeature_matcher.match(kps1, descriptors1, kps2, descriptors2)
    image = np.concatenate([image1, image2], axis=1)
    for kp1, kp2 in zip(matchkps1, matchkps2):
        cv2.circle(image, (int(kp1[0]+0.5), int(kp1[1]+0.5)), 2, (0, 0, 255), 1)
        cv2.circle(image, (int(kp2[0]+0.5+image1.shape[1]), int(kp2[1]+0.5)), 2, (0, 255, 0), 1)
        cv2.line(image, (int(kp1[0]+0.5), int(kp1[1]+0.5)), (int(kp2[0]+0.5+image1.shape[1]), int(kp2[1]+0.5)), (255, 255, 0))
    cv2.imwrite("test/flannsiftmatch.jpg", image)
    print("sift flann match done")