# -*- coding: utf-8 -*-
"""Multi-models generation based on categorizing feature"""

import numpy as np
from proj1_helpers import predict_labels
from cross_validation import k_fold_cross_validation

def multi_models_splitter(y, tx, feature_column_index, k, fun_model, fun_model_args):
    """Creates a predictions vector by creating different models based on the value
    of a categorizing feature in the dataset.

    Args:
        y (N x 1 vector): Labels vector.
        tx (N x D matrix): Features matrix (already pre-processed).
        feature_column_index (int): Categorizing feature's column index.
        k (int): Number of folds used for cross validation.
        fun_model (*function(...) return (weights,loss)): Function that computes a model.
        fun_model_args ([...]): Arguments list for fun_model (except y and tx).
    Returns:
        D x 1 vector: Predictions vector.
        float: Average of all predictions score.
        1 x F array: Array of categorization values.
        D x F matrix: Array of weights vector (one per categorizer value).
    """

    # feature_column_index must be positive
    if (feature_column_index < 0):
        raise ValueError("Parameter feature_column_index must be positive")

    # Get range of categorization values
    categorization_values = np.unique(tx[:,feature_column_index])
    num_models = len(categorization_values)

    # Number of datapoints
    data_size = tx.shape[0]

    # Initialize predictions vector
    y_pred = np.zeros(data_size)

    # Accumulators
    acc_weighted_pred_score = 0.0
    acc_weights = np.zeros( (tx.shape[1], num_models) )

    for i in range(0,num_models):
        
        # Only consider datapoints of one category
        idx_categorized = np.where( tx[:, feature_column_index] == categorization_values[i] )
        y_categorized = y[idx_categorized]
        tx_categorized = tx[idx_categorized]
        
        # Run cross-validation on the model
        avg_weights, avg_pred_score = k_fold_cross_validation(y_categorized, tx_categorized, k, fun_model, fun_model_args)

        # Update accumulators
        acc_weighted_pred_score += avg_pred_score * (tx_categorized.shape[0] / data_size)
        acc_weights[:,i] = np.copy(avg_weights)

        # Get predictions 
        y_pred_categorized = predict_labels(avg_weights, tx_categorized)

        # Get predictions back in correct order
        y_pred[idx_categorized] = y_pred_categorized

    return y_pred, acc_weighted_pred_score, categorization_values, acc_weights


def make_predictions_from_weights(tx, feature_column_index, weights, cat_values):
    """Creates a predictions vector based on a set of weights computed by multi_models_splitter.

    Args:
        tx (N x D matrix): Features matrix (already pre-processed).
        feature_column_index (int): Categorizing feature's column index.
        weights (D x F matrix): Array of weigths vector.
        cat_values (1 x F array): Array of categorization values.
    Returns:
        D x 1 vector: Predictions vector.
    """
    
    if (weights.shape[1] != cat_values.shape[0] ):
        raise ValueError("the number of columns of weight and the length of \
        categorization_values should be the same ")

    # Initialize predictions vector
    y_pred = np.zeros(tx.shape[0])

    for i in range(cat_values.shape[0]):
        
        # Only consider datapoints of one category
        idx_categorized = np.where( tx[:, feature_column_index] == cat_values[i] )
        tx_categorized = tx[idx_categorized]

        # Get predictions 
        y_pred_categorized = predict_labels(weights[:,i], tx_categorized)

        # Get predictions back in correct order
        y_pred[idx_categorized] = y_pred_categorized

    return y_pred



