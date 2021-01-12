import sys
sys.path.append("../../build/")
import pyVO
import numpy as np

random_sampler = pyVO.optim.RandomSampler(7)
random_sampler.Initialize(500)
sample = random_sampler.Sample()
print(sample)