# -*- coding: utf-8 -*-
"""Multi-models generation based on categorizing feature"""

import numpy as np
from proj1_helpers import predict_labels
from cross_validation import *

def multi_models_splitter(y_train, tx_train, tx_test, feature_column_index, k, fun_model, fun_model_args, fiveSeeds=False):
    """Creates a predictions vector by creating different models based on the value
    of a categorizing feature in the dataset.

    Args:
        y_train (N x 1 vector): Training labels vector.
        tx_train (N x D matrix): Training features matrix (already pre-processed).
        tx_test (N x D matrix): Test features matrix (already pre-processed).
        feature_column_index (int): Categorizing feature's column index.
        k (int): Number of folds used for cross validation.
        fun_model (*function(...) return (weights,loss)): Function that computes a model.
        fun_model_args ([...]): Arguments list for fun_model (except y and tx).
    Returns:
        D x 1 vector: Predictions vector for tx_test.
        float: Average of all predictions score.
    """

    # feature_column_index must be positive
    if (feature_column_index < 0):
        raise ValueError("Parameter feature_column_index must be positive")

    # Get range of categorization values
    categorization_values = np.unique(tx_train[:,feature_column_index])
    num_models = len(categorization_values)

    # Number of datapoints
    data_size = tx_train.shape[0]

    # Initialize predictions vector
    y_pred = np.zeros(tx_test.shape[0])

    # Accumulators
    acc_weighted_pred_score = 0.0

    for i in range(num_models):

        # Only consider datapoints of one category
        idx_categorized = np.where( tx_train[:, feature_column_index] == categorization_values[i] )
        y_categorized = y_train[idx_categorized]
        tx_categorized = tx_train[idx_categorized]

        if fiveSeeds:
            # Run cross-validation on the model
            weights, avg_pred_score = k_fold_cross_validation_5_seeds(y_categorized, tx_categorized, k, fun_model, fun_model_args)

        else:
            # Run cross-validation on the model
            weights, avg_pred_score = k_fold_cross_validation(y_categorized, tx_categorized, k, fun_model, fun_model_args)

        # Update accumulators
        acc_weighted_pred_score += avg_pred_score * (float(tx_categorized.shape[0]) / float(data_size))

        # Get predictions
        idx_categorized_test = np.where( tx_test[:, feature_column_index] == categorization_values[i] )
        tx_categorized_test = tx_test[idx_categorized_test]
        y_pred_categorized = predict_labels(weights, tx_categorized_test)

        # Get predictions back in correct order
        y_pred[idx_categorized_test] = y_pred_categorized

    return y_pred, acc_weighted_pred_score

def multi_models_splitter_experimental(y_train, tx_train, tx_test, feature_column_index, k, fun_model, fun_model_args):
    """Creates a predictions vector by creating different models based on the value
    of a categorizing feature in the dataset.

    Args:
        y_train (N x 1 vector): Training labels vector.
        tx_train (N x D matrix): Training features matrix (already pre-processed).
        tx_test (N x D matrix): Test features matrix (already pre-processed).
        feature_column_index (int): Categorizing feature's column index.
        k (int): Number of folds used for cross validation.
        fun_model (*function(...) return (weights,loss)): Function that computes a model.
        fun_model_args ([...]): Arguments list for fun_model (except y and tx).
    Returns:
        D x 1 vector: Predictions vector for tx_test.
        float: Average of all predictions score.
    """

    # feature_column_index must be positive
    if (feature_column_index < 0):
        raise ValueError("Parameter feature_column_index must be positive")

    # Get range of categorization values
    categorization_values = np.unique(tx_train[:,feature_column_index])
    num_models = len(categorization_values)

    # Accumulators
    idx_array = []
    y_pred_array = []
    pred_scores_array = []

    for i in range(num_models):

        # Only consider datapoints of one category
        idx_categorized = np.where( tx_train[:, feature_column_index] == categorization_values[i] )
        y_categorized = y_train[idx_categorized]
        tx_categorized = tx_train[idx_categorized]

        # Run cross-validation on the model
        weights, avg_pred_score = k_fold_cross_validation(y_categorized, tx_categorized, k, fun_model, fun_model_args)

        # Get predictions
        idx_categorized_test = np.where( tx_test[:, feature_column_index] == categorization_values[i] )
        tx_categorized_test = tx_test[idx_categorized_test]
        y_pred_categorized = predict_labels(weights, tx_categorized_test)

        # Update accumulators
        idx_array.append(idx_categorized_test)
        y_pred_array.append(y_pred_categorized)
        pred_scores_array.append(avg_pred_score)

    return idx_array, y_pred_array, pred_scores_array
