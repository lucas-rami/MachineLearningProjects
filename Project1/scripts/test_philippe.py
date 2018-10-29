import numpy as np
import proj1_helpers as helper
import data_preprocessing as preprocess
import multi_models_splitter as multi
import implementations as imp
import os


# Load training data  
y_train, tx_train, _ = helper.load_csv_data('../all/train.csv')

# Load test data
_, tx_test, ids = helper.load_csv_data('../all/test.csv')

preprocess.outliers_to_mean(tx_train)

degree = 6
processed_tx_train = preprocess.build_poly(tx_train, degree)[:,1:]

lambda_ = 10e-8
y_pred, pred_score, cat_values, weights = multi.multi_models_splitter(y_train, processed_tx_train, 22, 5, imp.ridge_regression, [lambda_])

print(pred_score)

# Remove outliers and do pre-processing
preprocess.outliers_to_mean(tx_test)
processed_tx_test = preprocess.build_poly(tx_test, degree)[:,1:]

# Use the multi_models_splitter function to compute our model 
y_pred_test = multi.make_predictions_from_weights(processed_tx_test, 22, weights, cat_values) 

# Save the predictions
program_path = os.path.dirname(os.path.realpath(__file__))
filename = program_path + '\\results\\philippe.csv'
helper.create_csv_submission(ids, y_pred_test, filename)