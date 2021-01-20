import sys
sys.path.append("../../build/")
import pyVO
import numpy as np
import math

class RansacPY(object):
    def __init__(self, estimator, supportmeasurer, sampler,
        max_error = 0.0, min_inlier_ratio = 0.1, confidence = 0.99,
        dyn_num_trials_multiplier = 3.0, min_num_trials = 0,
        max_num_trials = 10000000):
        super(RansacPY, self).__init__()

        self.estimator = estimator()
        self.kMinNumSamples = self.estimator.kMinNumSamples
        self.supportmeasurer = supportmeasurer()
        self.sampler = sampler(self.kMinNumSamples)
        self.max_error = max_error
        self.min_inlier_ratio = min_inlier_ratio
        self.confidence = confidence
        self.dyn_num_trials_multiplier = dyn_num_trials_multiplier
        self.min_num_trials = min_num_trials
        self.max_num_trials = max_num_trials

        self.check()

        # Determine max_num_trials based on assumed `min_inlier_ratio`.
        dyn_max_num_trials = self.compute_num_trials(int(100000 * self.min_inlier_ratio), 100000, self.confidence, self.dyn_num_trials_multiplier)
        self.max_num_trials = min(dyn_max_num_trials, self.max_num_trials)

    # check the ransac optional is leagle
    def check(self):
        assert self.max_error > 0.0
        assert self.min_inlier_ratio < 1.0 and self.min_inlier_ratio > 0.0
        assert self.confidence > 0.0 and self.confidence < 1.0
        assert self.min_num_trials < self.max_num_trials

    def sample(self, X, Y):
        X_rand = []
        Y_rand = []
        sample_idxes = self.sampler.Sample()
        for idx in sample_idxes:
            X_rand.append(X[idx])
            Y_rand.append(Y[idx])
        X_rand = np.array(X_rand).reshape(len(sample_idxes), -1)
        Y_rand = np.array(Y_rand).reshape(len(sample_idxes), -1)
        return X_rand, Y_rand

    # Determine the maximum number of trials required to sample at least one
    # outlier-free random set of samples with the specified confidence,
    # given the inlier ratio.
    
    # @param num_inliers				The number of inliers.
    # @param num_samples				The total number of samples.
    # @param confidence				Confidence that one sample is
    # 							outlier-free.
    # @param num_trials_multiplier   Multiplication factor to the computed
    # 						    number of trials.
    
    # @return               The required number of iterations.
    def compute_num_trials(self, num_inliers, num_samples, confidence,
        num_trials_multiplier):
        inlier_ratio = num_inliers / num_samples
        nom = 1 - confidence
        if nom <= 0:
            return 10000000
        denom = 1 - math.pow(inlier_ratio, self.kMinNumSamples)
        if denom <= 0:
            return 1
        return math.ceil(math.log(nom) / math.log(denom) * num_trials_multiplier)

    # Robustly estimate model with RANSAC (RANdom SAmple Consensus).
    #
    # @param X              Independent variables.
    # @param Y              Dependent variables.
    #
    # @return               The report with the results of the estimation.
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
                    dyn_max_num_trials = self.compute_num_trials(support.num_inliers, num_samples, self.confidence, self.dyn_num_trials_multiplier)
                if dyn_max_num_trials < num_trials and num_trials > self.min_num_trials:
                    abort = True
                    break
            num_trials += 1
        if support.num_inliers < self.kMinNumSamples:
            return success, num_trials, support, inlier_mask, model
        success = True
        residuals = self.estimator.Residuals(X, Y, model)
        assert len(residuals) == num_samples
        for i in range(num_samples):
            inlier_mask.append(residuals[i] <= max_residual)

        return success, num_trials, support, inlier_mask, model

if __name__ == "__main__":
    essential5_ransac = RansacPY(pyVO.estimator.EssentialMatrixFivePointEstimator, pyVO.optim.Support, pyVO.optim.RandomSampler, 0.1)
    # Check compute_num_trials
    print(essential5_ransac.compute_num_trials(1, 100, 0.99, 1.0) == 46051698048)
    print(essential5_ransac.compute_num_trials(10, 100, 0.99, 1.0) == 460515)
    print(essential5_ransac.compute_num_trials(10, 100, 0.999, 1.0) == 690773)
    print(essential5_ransac.compute_num_trials(10, 100, 0.999, 2.0) == 1381545)
    print(essential5_ransac.compute_num_trials(100, 100, 0.99, 1.0) == 1)
    print(essential5_ransac.compute_num_trials(100, 100, 0.999, 1.0) == 1)
    print(essential5_ransac.compute_num_trials(100, 100, 0.0, 1.0) == 1)