import numpy as np
import proj1_helpers as helper
import data_preprocessing as preprocess
import multi_models_splitter as multi
import implementations as imp
from cross_validation import k_fold_cross_validation
import os

# Load training data
y_train,tx_train,ids_train = helper.load_csv_data('../all/train.csv')

# Load test data
_,tx_test,ids = helper.load_csv_data('../all/test.csv')

# Seed the random number generator with a fixed value for consistent results
np.random.seed(20181028)

# Parameters
degrees = [3, 5, 6, 8, 10, 12]
lambdas = np.logspace(-9, 0, 10)
k_cross_val = [5]

# Best results
best_pred_score = 0.0
best_weights = 0
best_tx = 0

# Best parameters
best_degree = 0
best_lambda = 0.0
best_k = 0

# Normalizing data
preprocess.normalize_features(tx_train, tx_test)

print("Starting computations\n")

for degree in degrees: # For each degree...
    processed_tx_train = preprocess.build_poly(tx_train, degree)
    for lambda_ in lambdas: # For each lambda...
        for k in k_cross_val: # For each k...

            print("Trying (degree, lambda, k) = (" + str(degree) + ", " + str(lambda_) + ", " + str(k) + ")")

            # Use ridge_regression to compute our model
            weights, pred_score = k_fold_cross_validation(y_train, processed_tx_train, k, imp.ridge_regression, [lambda_])

            print("Got predictions score = " + str(pred_score) + "\n")

            if pred_score > best_pred_score:
                # Update best results
                best_weights = np.copy(weights)
                best_pred_score = pred_score

                # Update best parameters
                best_degree = degree
                best_lambda = lambda_
                best_k = k

print("Best score on training data is " + str(best_pred_score))
print("Best parameters are (degree, lambda, k) = (" + str(best_degree) + ", " + str(best_lambda) + ", " + str(best_k) + ")")

# Create the predictions
processed_tx_test = preprocess.build_poly(tx_test, best_degree)
y_pred = helper.predict_labels(best_weights, processed_tx_test)

# Save the predictions
program_path = os.path.dirname(os.path.realpath(__file__))
filename = program_path + '/results/run_ridge.csv'
helper.create_csv_submission(ids, y_pred, filename)

# Best score on training data is 0.817712
# Best parameters are (degree, lambda, k) = (12, 0.0001, 5)
