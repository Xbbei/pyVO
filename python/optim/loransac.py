import sys
sys.path.append("../../build/")
import pyVO
import numpy as np
import math
from ransac import RansacPY


# Implementation of LO-RANSAC (Locally Optimized RANSAC).
#
# "Locally Optimized RANSAC" Ondrej Chum, Jiri Matas, Josef Kittler, DAGM 2003.
class LoRansacPY(RansacPY):
    def __init__(self, estimator, loestimator, supportmeasurer, sampler,
        max_error = 0.0, min_inlier_ratio = 0.1, confidence = 0.99,
        dyn_num_trials_multiplier = 3.0, min_num_trials = 0,
        max_num_trials = 10000000):
        super(LoRansacPY, self).__init__(estimator, supportmeasurer, sampler, 
            max_error, min_inlier_ratio, confidence, dyn_num_trials_multiplier, min_num_trials, max_num_trials)
        self.loestimator = loestimator()

    def estimate(self, X, Y):
        assert X.shape[0] == Y.shape[0]
        num_samples = X.shape[0]
        success = False
        num_trials = 0
        support = pyVO.optim.Support()
        inlier_mask = []
        model = np.zeros([3, 3])

        if num_samples < self.kMinNumSamples:
            return success, num_trials, support, inlier_mask, model
        
        abort = False
        best_model_is_local = False
        max_residual = self.max_error * self.max_error
        self.sampler.Initialize(num_samples)

        max_num_trials = self.max_num_trials
        max_num_trials = min(max_num_trials, self.sampler.MaxNumSamples())
        dyn_max_num_trials = max_num_trials

        while (num_trials < max_num_trials):
            if abort:
                num_trials += 1
                break
            X_rand, Y_rand = self.sample(X, Y)
            sample_models = self.estimator.Estimate(X_rand, Y_rand)
            for sample_model in sample_models:
                residuals = self.estimator.Residuals(X, Y, sample_model)
                assert len(residuals) == num_samples
                support_tmp = self.supportmeasurer.Evaluate(residuals, max_residual)
                if self.supportmeasurer.Compare(support_tmp, support):
                    support = support_tmp
                    model = sample_model
                    best_model_is_local = False
                    if support.num_inliers > self.kMinNumSamples and support.num_inliers > self.loestimator.kMinNumSamples:
                        for local_num_trial in range(10):
                            X_inliers = []
                            Y_inliers = []
                            for i in range(num_samples):
                                if residuals[i] <= max_residual:
                                    X_inliers.append(X[i])
                                    Y_inliers.append(Y[i])
                            local_models = self.loestimator.Estimate(X_inliers, Y_inliers)
                            prev_best_num_inliers = support.num_inliers
                            best_local_residuals = None
                            for local_model in local_models:
                                residuals = self.loestimator.Residuals(X, Y, local_model)
                                assert len(residuals) == num_samples
                                local_support = self.supportmeasurer.Evaluate(residuals, max_residual)
                                if self.supportmeasurer.Compare(local_support, support):
                                    support = local_support
                                    model = local_model
                                    best_model_is_local = True
                                    best_local_residuals = residuals
                            if support.num_inliers <= prev_best_num_inliers:
                                break
                            residuals = best_local_residuals
                    dyn_max_num_trials = self.compute_num_trials(support.num_inliers, num_samples, self.confidence, self.dyn_num_trials_multiplier)
                if dyn_max_num_trials < num_trials and num_trials > self.min_num_trials:
                    abort = True
                    break
            num_trials += 1

        if support.num_inliers < self.kMinNumSamples:
            return success, num_trials, support, inlier_mask, model
        success = True
        if best_model_is_local:
            residuals = self.loestimator.Residuals(X, Y, model)
        else:
            residuals = self.estimator.Residuals(X, Y, model)
        assert len(residuals) == num_samples
        for i in range(num_samples):
            inlier_mask.append(residuals[i] <= max_residual)

        return success, num_trials, support, inlier_mask, model