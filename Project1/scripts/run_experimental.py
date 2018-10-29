import numpy as np
import proj1_helpers as helper
import data_preprocessing as preprocess
import multi_models_splitter as multi
import implementations as imp
import os

NB_MODELS = 4

# Load training data
y_train,tx_train,ids_train = helper.load_csv_data('../all/train.csv')

# Load test data
_,tx_test,ids = helper.load_csv_data('../all/test.csv')

# Seed the random number generator with a fixed value for consistent results
np.random.seed(20181029)

# Parameters
degrees = [1, 3, 5, 6, 8]
lambdas = np.logspace(-9, 0, 10)
k_cross_val = [5]

# Best results
best_idx = [0] * NB_MODELS
best_preds = [0] * NB_MODELS
best_preds_score = [0.0] * NB_MODELS

# Best parameters
best_degree = [0] * NB_MODELS
best_lambda = [0.0] * NB_MODELS
best_k = [0] * NB_MODELS

# Normalizing data
preprocess.normalize_features(tx_train, tx_test)

print("Starting computations\n")

for degree in degrees: # For each degree...
    processed_tx_train = preprocess.build_poly(tx_train, degree)
    processed_tx_test = preprocess.build_poly(tx_test, degree)
    for lambda_ in lambdas: # For each lambda...
        for k in k_cross_val: # For each k...

            print("Trying (degree, lambda, k) = (" + str(degree) + ", " + str(lambda_) + ", " + str(k) + ")")

            # Use the multi_models_splitter function to compute our model
            idx_array, y_pred_array, pred_scores_array = multi.multi_models_splitter_experimental(y_train, processed_tx_train, processed_tx_test, 23, k, imp.ridge_regression, [lambda_])

            for i in range(NB_MODELS):
                
                if pred_scores_array[i] > best_preds_score[i]:
                    # Update best results for this category
                    best_idx = idx_array
                    best_preds[i] = y_pred_array[i]
                    best_preds_score[i] = pred_scores_array[i]

                    # Update best parameters for this category
                    best_degree[i] = degree
                    best_lambda[i] = lambda_
                    best_k[i] = k

weighted_score = 0.0
for i in range(NB_MODELS):
    weighted_score += best_preds_score[i] * (float(best_preds[i].shape[0]) / float(tx_test.shape[0]) )

print("Best prediction score on training data is " + str(weighted_score))

for i in range(NB_MODELS):
    print("Best parameters for category " + str(i + 1) + " are (degree, lambda, k) = (" \
        + str(best_degree[i]) + ", " + str(best_lambda[i]) + ", " + str(best_k[i]) + ")")

# Create y_pred
y_pred = np.ones( (tx_test.shape[0], 1) )
for i in range(NB_MODELS):
    y_pred[best_idx[i]] = np.reshape(best_preds[i], (len(best_preds[i]), 1))

# Save the predictions
program_path = os.path.dirname(os.path.realpath(__file__))
filename = program_path + '/results/run_experimental.csv'
helper.create_csv_submission(ids, y_pred, filename)
