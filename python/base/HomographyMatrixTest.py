import sys
sys.path.append("../../build/")
import pyVO
import numpy as np

H = np.array([2.649157564634028, 4.583875997496426, 70.694447785121326,
              -1.072756858861583, 3.533262150437228, 1513.656999614321649,
              0.001303887589576, 0.003042206876298, 1], dtype=np.float64).reshape(3, 3) * 3.0

K = np.array([640, 0, 320, 0, 640, 240, 0, 0, 1], dtype=np.float64).reshape(3, 3)
Homography2Pose = pyVO.base.Homography2Pose()
Homography2Pose.DecomposeHomographyMatrix(H, K, K)
Rs = Homography2Pose.GetRotationVector()
ts = Homography2Pose.GetTranslationVector()
ns = Homography2Pose.GetNormalVector()

print(len(Rs) == 4)
print(len(ts) == 4)
print(len(ns) == 4)

R_ref = np.array([0.43307983549125, 0.545749113549648, -0.717356090899523,
      -0.85630229674426, 0.497582023798831, -0.138414255706431,
      0.281404038139784, 0.67421809131173, 0.682818960388909
 ], dtype=np.float64).reshape(3, 3)
t_ref = np.array([1.826751712278038, 1.264718492450820, 0.195080809998819], dtype=np.float64)
n_ref = np.array([-0.244875830334816, -0.480857890778889, -0.841909446789566], dtype=np.float64)

ref_solution_exists = False
for i in range(4):
    delta_R = Rs[i] - R_ref
    delta_t = ts[i] - t_ref
    delta_n = ns[i] - n_ref
    if np.linalg.norm(delta_R) < 1e-6 and \
        np.linalg.norm(delta_t) < 1e-6 and \
        np.linalg.norm(delta_n) < 1e-6:
        ref_solution_exists = True

print(ref_solution_exists)