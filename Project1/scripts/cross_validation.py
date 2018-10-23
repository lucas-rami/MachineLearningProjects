# -*- coding: utf-8 -*-
""" Cross validation """

import numpy as np
import implementations

def k_fold_cross_validation(k, y, tx, fun_make_model, fun_make_model_args):
    """Performs cross validation on the dataset.

    Args:
        k (int): Number of folds used for cross validation.
        y (N x 1 vector): Labels vector.
        tx (N x D matrix): Features matrix (already pre-processed).
        fun_make_model (*function(...) return (weights,loss)): Function that computes a model for the given features.
        fun_make_model_args ([...]): Arguments list for fun_make_model (except y and tx).
    Returns:
        float: Average of the k test errors
    """

    # k must be positive
    if (k <= 0):
        raise ValueError("Parameter k must be strictly positive")

    # Number of datapoints
    nb_data = tx.shape[0]

    # Accumulator for test error
    acc_test_error = 0.

    for i in range(k):

        # Create training and testing sets        
        indices = np.where(np.fromfunction(lambda x: x % k, (nb_data,), dtype=int) == i)
        indices_neg = np.where(np.fromfunction(lambda x: x % k, (nb_data,), dtype=int) != i)

        tx_training = tx[indices_neg] 
        tx_test = tx[indices]
        y_training = y[indices_neg]
        y_test = y[indices]

        # Compute our model
        weights, _ = fun_make_model(y_training, tx_training, *fun_make_model_args)

        # Compute the test error
        e = y_test - tx_test.dot(weights)
        test_error = implementations.compute_mse(e)

        # Accumulate the result
        acc_test_error += test_error

    return acc_test_error / k

