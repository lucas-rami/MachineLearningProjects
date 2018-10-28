# -*- coding: utf-8 -*-
"""Functions to pre-process data"""

import numpy as np

def build_poly(tx, degree):
    """Pre-processes the data by polynomial expansion. 
    
    Args:
        tx (N x D matrix): Features matrix (not pre-processed).
        degree (int): Expansion degree.
    Returns:
        (N x (1 + (D * degree)) matrix): Extended features matrix.
    """    
    poly = np.ones((len(tx), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(tx, deg)]
    return poly

def build_poly_diff_degrees(tx, degrees):
    """Pre-processes the data by polynomial expansion, with a different
    degree for each feature. 
    
    Args:
        tx (N x D matrix): Features matrix (not pre-processed).
        degrees (int): Expansion degree for each feature.
    Returns:
        Extended features matrix.
    """
    if len(tx[0]) != len(degrees):
        raise ValueError("Array of degrees must be of same length that columns of data")

    poly_diff = np.ones((len(tx), 1))
    for ind, degree in enumerate(degrees):
        poly = build_poly(tx[:, ind], int(degree))
        poly_diff = np.append(poly_diff, poly[:, 1:], axis=1)
    
    return poly_diff

def outliers_to_mean(tx):
    """Pre-processes the data by replacing outliers value (-999) with
    mean of corresponding feature. 
    
    Args:
        tx (N x D matrix): Features matrix (not pre-processed).
    Returns:
        nothing
    """

    for i in range(0,tx.shape[1]):
        # Compute the mean for each column (removing outliers)
        feature_mean = tx[:,i][np.where(tx[:,i] != -999)].mean()
        # Replace outliers value with mean
        tx[:,i][tx[:,i] == -999] = feature_mean
