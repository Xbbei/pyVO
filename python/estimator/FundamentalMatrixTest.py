import sys
sys.path.append("../../build/")
import pyVO
import numpy as np

# 5 points algorithm test
points1 = np.array([0.4964, 1.0577,  0.3650,  -0.0919, -0.5412,
    0.0159, -0.5239, 0.9467,  0.3467,  0.5301,
    0.2797, 0.0012,  -0.1986, 0.0460]).reshape(-1, 2)

points2 = np.array([0.7570, 2.7340,  0.3961,  0.6981, -0.6014,
    0.7110, -0.7385, 2.2712,  0.4177, 1.2132,
    0.3052, 0.4835,  -0.2171, 0.5057]).reshape(-1, 2)

sevenpointestimator = pyVO.estimator.FundamentalMatrixSevenPointEstimator()
gt = np.array([4.81441976, -8.16978909, 6.73133404, 5.16247992, 0.19325606, -2.87239381, -9.92570126, 3.64159554, 1.0]).reshape(3, 3)
pr = sevenpointestimator.Estimate(points1, points2)
print(np.isclose(pr[0], gt, atol=1e-6))

# 8 points algorithm test
points1 = np.array([1.839035, 1.924743, 0.543582,  0.375221,
    0.473240, 0.142522, 0.964910,  0.598376,
    0.102388, 0.140092, 15.994343, 9.622164,
    0.285901, 0.430055, 0.091150,  0.254594]).reshape(-1, 2)

points2 = np.array([1.002114, 1.129644, 1.521742, 1.846002, 
    1.084332, 0.275134, 0.293328, 0.588992, 
    0.839509, 0.087290, 1.779735, 1.116857,
    0.878616, 0.602447, 0.642616, 1.028681,]).reshape(-1, 2)

eightpointestimator = pyVO.estimator.FundamentalMatrixEightPointEstimator()
gt = np.array([-0.217859, 0.419282, -0.0343075, -0.0717941, 0.0451643, 0.0216073, 0.248062, -0.429478, 0.0221019]).reshape(3, 3)
pr = eightpointestimator.Estimate(points1, points2)
print(np.isclose(pr[0], gt, atol=1e-5))