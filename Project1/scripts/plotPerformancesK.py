import numpy as np
import proj1_helpers as helper
import data_preprocessing as preprocess
import multi_models_splitter as multi
import implementations as imp
import os
import visualization as vis

print("Loading training data...")
# Load training data
y_train,tx_train,ids_train = helper.load_csv_data('../all/train.csv')
print("Done!")

print("Loading test data...")
# Load test data
_,tx_test,ids = helper.load_csv_data('../all/test.csv')
print("Done!")

# Seed the random number generator with a fixed value for consistent results
np.random.seed(20181028)

# Parameters
degrees = list(range(11, 12))
lambdas = np.logspace(-4.5, -4.5, 1)
k_cross_val = list(range(2, 10))
score = [0]*len(k_cross_val)

# Best results
best_pred_score = 0.0
best_pred = np.array( (tx_test.shape[0], 1) )

# Best parameters
best_degree = 0
best_lambda = 0.0
best_k = 0

# Normalizing data
preprocess.normalize_features(tx_train, tx_test)

print("Starting computations\n")

for ind_degree, degree in enumerate(degrees): # For each degree...
    processed_tx_train = preprocess.build_poly(tx_train, degree)
    processed_tx_test = preprocess.build_poly(tx_test, degree)
    for ind_lambda, lambda_ in enumerate(lambdas): # For each lambda...
        for ind_k, k in enumerate(k_cross_val): # For each k...

            print("Trying (degree, lambda, k) = (" + str(degree) + ", " + str(lambda_) + ", " + str(k) + ")")

            # Use the multi_models_splitter function to compute our model
            y_pred, pred_score = multi.multi_models_splitter(y_train, processed_tx_train, processed_tx_test, 23, k, imp.ridge_regression, [lambda_], fiveSeeds=True)

            print("Got predictions score = " + str(pred_score) + "\n")

            if pred_score > best_pred_score:
                # Update best results
                best_pred_score = pred_score
                best_pred = y_pred

                # Update best parameters
                best_degree = degree
                best_lambda = lambda_
                best_k = k
            score[ind_k] = pred_score

np.savetxt('k_score.out', (score, k_cross_val))

vis.cross_validation_visualization_k(k_cross_val, score)

print("Best prediction score on training data is " + str(best_pred_score))
print("Best parameters are (degree, lambda, k) = (" + str(best_degree) + ", " + str(best_lambda) + ", " + str(best_k) + ")")
