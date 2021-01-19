import sys
sys.path.append("../../build/")
import pyVO
import numpy as np

image = np.zeros([256, 256, 1], dtype=np.uint8)
for r in range(128 - 32, 128 + 32):
    for c in range(128 - 32, 128 + 32):
        image[r, c, 0] = 255

orb_extract_options = pyVO.feature.ORBExtractionOptions()
orb_extract_options.nFeatures = 1000
orb_extract = pyVO.feature.ORBExtract(orb_extract_options)
print(orb_extract.ExtractORBFeatures(image))
keypoints = orb_extract.getKeyPoints()
descriptors = orb_extract.getDescriptors()

print(keypoints)
print(np.array(descriptors).shape)