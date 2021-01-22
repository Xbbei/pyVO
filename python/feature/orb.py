import sys
sys.path.append("../../build/")
import pyVO
import numpy as np
import cv2

class ORBExtractor(object):
    def __init__(self, nFeatures=8192, scaleFactor=1.2, nLevels=8, iniThFAST=20, minThFAST=7):
        super(ORBExtractor, self).__init__()

        self.orb_extract_options = pyVO.feature.ORBExtractionOptions()
        self.orb_extract_options.nFeatures = nFeatures
        self.orb_extract_options.scaleFactor = scaleFactor
        self.orb_extract_options.nLevels = nLevels
        self.orb_extract_options.iniThFAST = iniThFAST
        self.orb_extract_options.minThFAST = minThFAST

        self.orb_extract = pyVO.feature.ORBExtract(self.orb_extract_options)

    def extract(self, image):
        assert len(image.shape) == 3
        if image.shape[2] > 1:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        self.orb_extract.ExtractORBFeatures(gray)
        keypoints = self.orb_extract.getKeyPoints()
        kps = []
        for kp in keypoints:
            kps.append([kp.x, kp.y])
        kps = np.array(kps)
        descriptors = self.orb_extract.getDescriptors()
        descriptors = np.array(descriptors)
        return kps, descriptors

if __name__ == "__main__":
    orb_extractor = ORBExtractor()
    image = cv2.imread("data/test1.jpg", -1)
    kps, descriptors = orb_extractor.extract(image)

    for kp in kps:
        x, y = kp[0], kp[1]
        cv2.circle(image, (int(x+0.5), int(y+0.5)), 2, (0, 0, 255), 1)

    cv2.imwrite("test/orbtest.jpg", image)