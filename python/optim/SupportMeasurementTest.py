import sys
sys.path.append("../../build/")
import pyVO
import numpy as np

support = pyVO.optim.Support()
print(support, support.num_inliers, support.residual_sum)

measurer = pyVO.optim.InlierSupportMeasurer()
residuals = [-1.0, 0.0, 1.0, 2.0]
support1 = measurer.Evaluate(residuals, 1.0)
print(support1.num_inliers == 3)
print(support1.residual_sum == 0.0)