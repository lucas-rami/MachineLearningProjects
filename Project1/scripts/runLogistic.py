import numpy as np
import proj1_helpers as helper
import data_preprocessing as preprocess
import multi_models_splitter as multi
import implementations as imp
import os

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
degrees = list(range(1, 2))
lambdas = np.logspace(-9, 0, 10)
k_cross_val = [5]

# Best results
best_pred_score = 0.0
best_pred = np.array( (tx_test.shape[0], 1) )

# Best parameters
best_degree = 0
best_lambda = 0.0
best_k = 0
max_iters = 2001
gamma = 1e-7

# Normalizing data
preprocess.normalize_features(tx_train, tx_test)

print("Starting computations\n")

for degree in degrees: # For each degree...
    processed_tx_train = preprocess.build_poly(tx_train, degree)
    processed_tx_test = preprocess.build_poly(tx_test, degree)
    initial_w = np.ones(processed_tx_train.shape[1])
    for lambda_ in lambdas: # For each lambda...
        for k in k_cross_val: # For each k...

            print("Trying (degree, lambda, k) = (" + str(degree) + ", " + str(lambda_) + ", " + str(k) + ")")

            # Use the multi_models_splitter function to compute our model
            y_pred, pred_score = multi.multi_models_splitter(y_train, processed_tx_train, processed_tx_test, 23, k, imp.logistic_regression, [initial_w, max_iters, gamma])

            print("Got predictions score = " + str(pred_score) + "\n")

            if pred_score > best_pred_score:
                # Update best results
                best_pred_score = pred_score
                best_pred = y_pred

                # Update best parameters
                best_degree = degree
                best_lambda = lambda_
                best_k = k

print("Best prediction score on training data is " + str(best_pred_score))
print("Best parameters are (degree, lambda, k) = (" + str(best_degree) + ", " + str(best_lambda) + ", " + str(best_k) + ")")

# Save the predictions
program_path = os.path.dirname(os.path.realpath(__file__))
filename = program_path + '/results/run_cross2.csv'
helper.create_csv_submission(ids, best_pred, filename)
