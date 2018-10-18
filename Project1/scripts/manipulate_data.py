# -*- coding: utf-8 -*-
"""Functions to manipulate the data"""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    mx = np.ones((x.shape[0], (x.shape[1]*degree)+1))
    for j in range(1, degree+1):
        mx[:, x.shape[1]*(j-1)+1:x.shape[1]*j+1] = np.power(x, j)
    return mx