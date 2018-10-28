# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # ridge regression: TODO
    # ***************************************************
    b = (tx).T.dot(y)
    Ridge = 2 * len(y) * lambda_ * np.identity(len(b))
    A = (tx).T.dot(tx)+Ridge
    return np.linalg.solve(A, b)
