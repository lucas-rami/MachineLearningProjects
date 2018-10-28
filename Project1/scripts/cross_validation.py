# -*- coding: utf-8 -*-
"""Cross validation"""

import numpy as np
import implementations
import proj1_helpers

def k_fold_cross_validation(y, tx, k, nb_it, fun_model, fun_model_args=[]):
    """Performs cross validation on the dataset.

    Args:
        y (N x 1 vector): Labels vector.
        tx (N x D matrix): Features matrix (already pre-processed).
        k (int): Number of folds used for cross validation.
        nb_it (int): Number of iterations of k-fold
        fun_model (*function(...) return (weights,loss)): Function that computes a model.
        fun_model_args ([...]): Arguments list for fun_model (except y and tx).
    Returns:
        D x 1 vector: Average of all weights.
        float: Average of all predictions score.
    """

    # k must be positive
    if (k <= 0):
        raise ValueError("Parameter k must be strictly positive")

    # nb_it must >= 1
    if (k < 1):
        raise ValueError("Parameter nb_it must be >= 1")

    # Number of datapoints
    data_size = tx.shape[0]

    # Accumulators
    acc_pred_score = 0.0
    acc_weights = np.zeros( (tx.shape[1], k) )

    # Seed the random number generator with a fixed value for consistent results
    np.random.seed(20181028)

    for _ in range(nb_it):

        # Create random partioning of data
        shuffle_indices = np.random.permutation(np.arange(data_size))

        for i in range(k):

            start_index = i * k
            end_index = min((i + 1) * k, data_size)

            # Test data for this iteration
            y_test = y[ shuffle_indices[start_index:end_index] ]
            tx_test = tx[ shuffle_indices[start_index:end_index] ]

            # train data for this iteration
            indices_train = np.concatenate( (shuffle_indices[:start_index], shuffle_indices[end_index:]) , axis=0)
            tx_train = tx[ indices_train ]
            y_train = y[ indices_train ]

            # Compute our model
            weights, _ = fun_model(y_train, tx_train, *fun_model_args)

            # Compute the predictions score
            pred_score = compute_predictions_score(y_test, weights, tx_test)

            # Accumulate the results
            acc_pred_score += pred_score
            acc_weights[:,i] = weights

    # Average the weights and test errors
    avg_weights = acc_weights.sum(axis=1) / (k * nb_it)
    acc_pred_score /= (k * nb_it)  

    return avg_weights, acc_pred_score

def compute_predictions_score(y_ref, weights, data):
    """Computes the prediction score obtained by a weights vector.

    Args:
        y_ref (N x 1 vector): Reference labels vector.
        weights (D x 1 matrix): Weights vector
        data (N x D matrix): Features matrix (already pre-processed).
    Returns:
        float: the proportion of correctly predicted labels (between 0 and 1)
    """
    y_pred = proj1_helpers.predict_labels(weights, data)
    return float(np.sum(y_pred == y_ref)) / float(y_ref.shape[0])



