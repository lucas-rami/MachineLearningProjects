import numpy as np
import proj1_helpers as helper
import data_preprocessing as preprocess
import multi_models_splitter as multi
import implementations as imp
import os
import visualization as vis

# Parameters
degrees = list(range(9, 16))
lambdas = np.logspace(-9, 0, 10)
score = [0]*len(degrees)
k_cross_val = [5]
np.random.seed(1)

for ind_degree, degree in enumerate(degrees): # For each degree...
    score[ind_degree] = np.random.random()


np.savetxt('degree_score.out', (score, degrees))

vis.cross_validation_visualization(degrees, score)
