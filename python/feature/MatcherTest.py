import sys
sys.path.append("../../build/")
import pyVO
import numpy as np
import cv2
sys.path.append("../optim/")
from ransac import RansacPY

orb_extract_options = pyVO.feature.ORBExtractionOptions()
# orb_extract_options.nFeatures = 1000
orb_extract = pyVO.feature.ORBExtract(orb_extract_options)

image1 = cv2.imread("test1.jpg", -1)
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
orb_extract.ExtractORBFeatures(gray1)
image_keypoints1 = orb_extract.getKeyPoints()
image_descriptors1 = orb_extract.getDescriptors()
# print(image_keypoints1, image_descriptors1)

image2 = cv2.imread("test2.jpg", -1)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
orb_extract.ExtractORBFeatures(gray2)
image_keypoints2 = orb_extract.getKeyPoints()
image_descriptors2 = orb_extract.getDescriptors()
# print(image_keypoints2, image_descriptors2)
print("feature extract done")

# sift_extract_options = pyVO.feature.SiftExtractionOptions()
# sift_extract_options.use_gpu = False
# sift_extract = pyVO.feature.SiftExtract(sift_extract_options)

# image1 = cv2.imread("test1.jpg", -1)
# gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
# sift_extract.ExtractSiftFeatures(gray1)
# image_keypoints1 = sift_extract.getKeyPoints()
# image_descriptors1 = sift_extract.getDescriptors()
# # print(image_keypoints1, image_descriptors1)

# image2 = cv2.imread("test2.jpg", -1)
# gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
# sift_extract.ExtractSiftFeatures(gray2)
# image_keypoints2 = sift_extract.getKeyPoints()
# image_descriptors2 = sift_extract.getDescriptors()
# print("feature extract done")

# BF
# matches = pyVO.feature.BFComputeFeatureMatches(image_descriptors1, image_descriptors2, "orb")
# matches = pyVO.feature.BFComputeFeatureMatches(image_descriptors1, image_descriptors2, "sift")

# FLANN
matches = pyVO.feature.FLANNComputeFeatureMatches(image_descriptors1, image_descriptors2, "orb")
# matches = pyVO.feature.FLANNComputeFeatureMatches(image_descriptors1, image_descriptors2, "sift")

image = np.concatenate([image1, image2], axis=1)
X = []
Y = []
for match in matches:
    if match[0].distance / match[1].distance > 0.8:
        continue
    first = match[0].point2D_idx1
    second = match[0].point2D_idx2
    X.append([image_keypoints1[first].x, image_keypoints1[first].y])
    Y.append([image_keypoints2[second].x, image_keypoints2[second].y])
    circle1 = (image_keypoints1[first].x, image_keypoints1[first].y)
    circle2 = (image_keypoints2[second].x + image1.shape[1], image_keypoints2[second].y)
    cv2.circle(image, (int(circle1[0]+0.5), int(circle1[1]+0.5)), 2, (0, 0, 255), 1)
    cv2.circle(image, (int(circle2[0]+0.5), int(circle2[1]+0.5)), 2, (0, 255, 0), 1)
    cv2.line(image, (int(circle1[0]+0.5), int(circle1[1]+0.5)), (int(circle2[0]+0.5), int(circle2[1]+0.5)), (255, 255, 0))
cv2.imwrite("concat.jpg", image)

print("flann match done")
X = np.array(X)
Y = np.array(Y)
essential5_ransac = RansacPY(pyVO.estimator.EssentialMatrixFivePointEstimator, pyVO.optim.InlierSupportMeasurer, pyVO.optim.RandomSampler, 2.0)
success, num_trials, support, inlier_mask, model = essential5_ransac.estimate(X, Y)
print(success)
print(model)

image = np.concatenate([image1, image2], axis=1)
for i, (px, py) in enumerate(zip(X, Y)):
    if not inlier_mask[i]:
        continue
    circle1 = px
    circle2 = [py[0] + image1.shape[1], py[1]]
    cv2.circle(image, (int(circle1[0]+0.5), int(circle1[1]+0.5)), 2, (0, 0, 255), 1)
    cv2.circle(image, (int(circle2[0]+0.5), int(circle2[1]+0.5)), 2, (0, 255, 0), 1)
    cv2.line(image, (int(circle1[0]+0.5), int(circle1[1]+0.5)), (int(circle2[0]+0.5), int(circle2[1]+0.5)), (255, 255, 0))
cv2.imwrite("ransac_concat.jpg", image)

# sift_extract_options = pyVO.feature.SiftExtractionOptions()
# sift_extract_options.use_gpu = False
# sift_extract = pyVO.feature.SiftExtract(sift_extract_options)

# image1 = cv2.imread("test1.jpg", -1)
# gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
# sift_extract.ExtractSiftFeatures(gray1)
# image_keypoints1 = sift_extract.getKeyPoints()
# image_descriptors1 = sift_extract.getDescriptors()
# # print(image_keypoints1, image_descriptors1)

# image2 = cv2.imread("test2.jpg", -1)
# gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
# sift_extract.ExtractSiftFeatures(gray2)
# image_keypoints2 = sift_extract.getKeyPoints()
# image_descriptors2 = sift_extract.getDescriptors()
# print("feature extract done")

# # FLANN
# matches = pyVO.feature.FLANNComputeFeatureMatches(image_descriptors1, image_descriptors2, pyVO.feature.SiftDescriptorDistance)

# image = np.concatenate([image1, image2], axis=1)
# for match in matches:
#     if match[0].distance / match[1].distance > 0.2:
#         continue
#     first = match[0].point2D_idx1
#     second = match[0].point2D_idx2
#     circle1 = (image_keypoints1[first].x, image_keypoints1[first].y)
#     circle2 = (image_keypoints2[second].x + image1.shape[1], image_keypoints2[second].y)
#     cv2.circle(image, (int(circle1[0]+0.5), int(circle1[1]+0.5)), 2, (0, 0, 255), 1)
#     cv2.circle(image, (int(circle2[0]+0.5), int(circle2[1]+0.5)), 2, (0, 255, 0), 1)
#     cv2.line(image, (int(circle1[0]+0.5), int(circle1[1]+0.5)), (int(circle2[0]+0.5), int(circle2[1]+0.5)), (255, 255, 0))
# cv2.imwrite("concat.jpg", image)