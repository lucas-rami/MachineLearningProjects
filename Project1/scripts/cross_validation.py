# -*- coding: utf-8 -*-
""" Cross validation """

import numpy as np

def k_fold_cross_validation(k, y, tx, fun_make_model, fun_make_model_args):
    """Performs cross validation on the dataset.

    Args:
        k (int): Number of folds used for cross validation.
        y (N x 1 vector): Labels vector.
        tx (N x D matrix): Features matrix (already pre-processed).
        fun_make_model (*function(...) return (weights,loss)): Function that computes a model for the given features.
        fun_make_model_args ([...]): Arguments list for fun_make_model.

    Returns:
        float: Average of the k test errors
    """

    # k must be positive
    if (k <= 0):
        return float(-1)

    # Number of datapoints
    nb_datapoints = tx.shape[0]

    for i in range(k):
        np.fromfunction(lambda x: x if x % k == i else -1, (nb,), dtype=int)



    raise NotImplementedError