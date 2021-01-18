import sys
sys.path.append("../../build/")
import pyVO
import numpy as np

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