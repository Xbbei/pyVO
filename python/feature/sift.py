import sys
sys.path.append("../../build/")
import pyVO
import numpy as np
import cv2

class SiftExtractor(object):
    def __init__(self, max_image_size=3200, max_num_features=8192, num_octaves=4,
        first_octave=-1, octave_resolution=3, edge_threshold=10.0, estimate_affine_shape=False,
        max_num_orientations=2, upright=False, darkness_adaptivity=False, domain_size_pooling=False,
        dsp_min_scale=1.0/6.0, dsp_max_scale=3.0, dsp_num_scales=10, use_gpu=False,
        gpu_index="-1", peak_threshold=0.02/3.0):
        super(SiftExtractor, self).__init__()

        self.sift_extract_options = pyVO.feature.SiftExtractionOptions()
        self.sift_extract_options.max_image_size = max_image_size
        self.sift_extract_options.max_num_features = max_num_features
        self.sift_extract_options.num_octaves = num_octaves
        self.sift_extract_options.first_octave = first_octave
        self.sift_extract_options.octave_resolution = octave_resolution
        self.sift_extract_options.edge_threshold = edge_threshold
        self.sift_extract_options.estimate_affine_shape = estimate_affine_shape
        self.sift_extract_options.max_num_orientations = max_num_orientations
        self.sift_extract_options.upright = upright
        self.sift_extract_options.darkness_adaptivity = darkness_adaptivity
        self.sift_extract_options.domain_size_pooling = domain_size_pooling
        self.sift_extract_options.dsp_min_scale = dsp_min_scale
        self.sift_extract_options.dsp_max_scale = dsp_max_scale
        self.sift_extract_options.dsp_num_scales = dsp_num_scales
        self.sift_extract_options.use_gpu = use_gpu
        self.sift_extract_options.gpu_index = gpu_index
        self.sift_extract_options.peak_threshold = peak_threshold

        self.sift_extract = pyVO.feature.SiftExtract(self.sift_extract_options)

    def extract(self, image):
        assert len(image.shape) == 3
        if image.shape[2] > 1:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        self.sift_extract.ExtractSiftFeatures(gray)
        keypoints = self.sift_extract.getKeyPoints()
        kps = []
        for kp in keypoints:
            kps.append([kp.x, kp.y])
        kps = np.array(kps)
        descriptors = self.sift_extract.getDescriptors()
        descriptors = np.array(descriptors)
        return kps, descriptors

if __name__ == "__main__":
    sift_extractor = SiftExtractor()
    image = cv2.imread("data/test1.jpg", -1)
    kps, descriptors = sift_extractor.extract(image)

    for kp in kps:
        x, y = kp[0], kp[1]
        cv2.circle(image, (int(x+0.5), int(y+0.5)), 2, (0, 0, 255), 1)

    cv2.imwrite("test/sifttest.jpg", image)