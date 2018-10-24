# -*- coding: utf-8 -*-
"""Functions to manipulate the data"""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly

def build_poly_diff_degrees(x, degrees):
    """"""
    if len(x[0]) != len(degrees):
        raise ValueError("Array of degrees must be of same length that columns of data")

    poly_diff = np.ones((len(x), 1))
    for ind, degree in enumerate(degrees):
        poly = build_poly(x[:, ind], int(degree))
        poly_diff = np.append(poly_diff, poly[:, 1:], axis=1)
    
    return poly_diff