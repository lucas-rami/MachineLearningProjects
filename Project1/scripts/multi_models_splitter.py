# -*- coding: utf-8 -*-
"""A function to split the data and work with multiple models
"""
import numpy as np
from proj1_helpers import predict_labels
from cross_validation import k_fold_cross_validation

def multi_model_splitter(y, tx, feature_column_index, k, fun_model, fun_model_args):
    """Performs cross validation on the dataset.

    Args:
        y (N x 1 vector): Labels vector.
        tx (N x D matrix): Features matrix (already pre-processed).
        feature_column_index (int): Classifier feature index
        k (int): Number of folds used for cross validation.
        fun_model (*function(...) return (weights,loss)): Function that computes a model.
        fun_model_args ([...]): Arguments list for fun_model (except y and tx).
    Returns:
        D x 1 vector: Predictions vector
        float: Average of all prediction errors.
    """

    # Get range of categorization values
    categorization_values = np.unique(tx[:,feature_column_index])
    num_models = len(categorization_values)

    # Number of datapoints
    data_size = tx.shape[0]

    # Initialize predictions vector
    y_pred = np.zeros(data_size)
    acc_weighted_pred_error = 0.0

    for i in range(0,num_models):
        
        # Only consider datapoints of one category
        idx_categorized = np.where( tx[:, feature_column_index] == categorization_values[i] )
        y_categorized = y[idx_categorized]
        tx_categorized = tx[idx_categorized]
        
        # Run cross-validation on the model
        avg_weights, avg_pred_error = k_fold_cross_validation(y_categorized, tx_categorized, k, fun_model, fun_model_args)

        # Accumulate prediction error
        acc_weighted_pred_error += avg_pred_error * (tx_categorized.shape[0] / data_size)

        # Get predictions 
        y_pred_categorized = predict_labels(avg_weights, tx_categorized)

        # Get predictions back in correct order
        y_pred[idx_categorized] = y_pred_categorized

    return y_pred, acc_weighted_pred_error
