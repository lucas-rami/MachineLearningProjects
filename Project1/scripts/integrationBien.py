import numpy as np
import proj1_helpers as helper
import data_preprocessing as preprocess
import multi_models_splitter as multi
import implementations as imp
import os

# Load training data
y_train,tx_train,ids_train = helper.load_csv_data('../all/train.csv')

# Load test data
_,tx_test,ids = helper.load_csv_data('../all/test.csv')

# Seed the random number generator with a fixed value for consistent results
np.random.seed(20181028)

# Parameters
degrees = [6]
lambdas = [10e-8]  #np.logspace(-6, 0, 13)
k_cross_val = [5]

# Best results
best_pred_score = 0.0
best_pred = np.array( (tx_test.shape[0], 1) )

# Best parameters
best_degree = 0
best_lambda = 0.0
best_k = 0

# Normalizing data
#preprocess.outliers_to_mean(tx_train)
#preprocess.outliers_to_mean(tx_test)

tx_train_modified = np.copy(tx_train)
tx_test_modified = np.copy(tx_test)
for i in range(0,tx_train.shape[1]):
    feature_mean_train = tx_train[:,i][np.where(tx_train[:,i] != -999)].mean()
    feature_mean_test = tx_train[:,i][np.where(tx_train[:,i] != -999)].mean()
    tx_train_modified[:,i][tx_train[:,i] == -999] = feature_mean_train
    tx_test_modified[:,i][tx_test[:,i] == -999] = feature_mean_test

print("Starting computations\n")

for degree in degrees: # For each degree...
    processed_tx_train = preprocess.build_poly(tx_train_modified, degree)
    processed_tx_test = preprocess.build_poly(tx_test_modified, degree)
    for lambda_ in lambdas: # For each lambda...
        for k in k_cross_val: # For each k...

            print("Trying (degree, lambda, k) = (" + str(degree) + ", " + str(lambda_) + ", " + str(k) + ")")

            # Use the multi_models_splitter function to compute our model
            y_pred, pred_score = multi.multi_models_splitter_complete(y_train, processed_tx_train, processed_tx_test, 23, k, imp.ridge_regression, [lambda_])

            print("Got score = " + str(pred_score))

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

y_pred2 = multi.multi_model_datasets_creation(y_train,tx_train_modified,tx_test_modified,22,preprocess.build_poly,degrees,imp.ridge_regression,lambdas)
yDiff = sum(abs(best_pred-y_pred2))

# Save the predictions
program_path = os.path.dirname(os.path.realpath(__file__))
filename = program_path + '/results/integration1.csv'
helper.create_csv_submission(ids, best_pred, filename)
filename2 = program_path + '/results/integration1_phil.csv'
helper.create_csv_submission(ids, y_pred2, filename2)
