import sys
sys.path.append("../build/")
import pyVO
import numpy as np

homography_estimator = pyVO.homography_matrix.HomographyMatrixEstimator()
for x in range(10):
    H0 = np.array([x, 0.2, 0.3, 30, 0.2, 0.1, 0.3, 20, 1], dtype=np.float64).reshape(3, 3)

    src = []
    src.append(np.array([x, 0], dtype=np.float64))
    src.append(np.array([1, 0], dtype=np.float64))
    src.append(np.array([2, 1], dtype=np.float64))
    src.append(np.array([10, 30], dtype=np.float64))

    dst = []
    for i in range(4):
        tmp = np.array([src[i][0], src[i][1], 1.0], dtype=np.float64)
        tmp = H0.dot(tmp)
        dst.append(np.array([tmp[0] / tmp[2], tmp[1] / tmp[2]], dtype=np.float64))

    src_np = np.array(src).reshape(-1, 2)
    dst_np = np.array(dst).reshape(-1, 2)
    models = homography_estimator.Estimate(src_np, dst_np)

    residuals = homography_estimator.Residuals(src_np, dst_np, models[0])
    for residual in residuals:
        print(residual < 1e-6)