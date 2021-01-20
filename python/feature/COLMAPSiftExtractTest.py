import sys
sys.path.append("../../build/")
import pyVO
import numpy as np
import cv2

image = np.zeros([256, 256, 1], dtype=np.uint8)
for r in range(128 - 32, 128 + 32):
    for c in range(128 - 32, 128 + 32):
        image[r, c, 0] = 255

sift_extract_options = pyVO.feature.SiftExtractionOptions()
sift_extract_options.use_gpu = False
sift_extract = pyVO.feature.SiftExtract(sift_extract_options)
print(sift_extract.ExtractSiftFeatures(image))
keypoints = sift_extract.getKeyPoints()
descriptors = sift_extract.getDescriptors()

print(keypoints)
print(np.array(descriptors).shape)

image = cv2.imread("test1.jpg", -1)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
sift_extract.ExtractSiftFeatures(gray)
image_keypoints = sift_extract.getKeyPoints()
image_descriptors = sift_extract.getDescriptors()

for kp in image_keypoints:
    x, y = kp.x, kp.y
    cv2.circle(image, (int(x+0.5), int(y+0.5)), 2, (0, 0, 255), 1)

cv2.imwrite("sifttest.jpg", image)