# -*- coding: utf-8 -*-
""" Finding best polynomial extension for the input data """

import numpy as np
from implementations import *
from manipulate_data import *
from cross_validation import *

def find_best_poly(k, y, tx, fun_make_model, max_degree, lambdas):

    # Initializing parameters
    mx_best = np.ones((len(tx), 1))
    degrees_best = np.ones((len(tx[0]), 1))

    for i in range(tx.shape[1]):
        # Optimizing each feature
        te_best = float('inf')
        te_best_lambda = float('inf')
        print("Optimizing feature", i+1, "/", tx.shape[1])
        for j in range(max_degree):
            # Optimizing the degree of expansion for feature i
            mx = build_poly(tx[:, i], j)
            for lambda_ in lambdas:
                # Optimizing the lambda for feature i with expansion of degree j
                # Calculating the test error and corresponding weights
                te_tmp, weights_tmp = k_fold_cross_validation(k, y, mx, fun_make_model, lambda_)
                if te_tmp < te_best_lambda:
                    # Updating the best test error and weights as a function of lambda
                    te_best_lambda = te_tmp
                    weights_best_lambda = weights_tmp
                    best_lambda = lambda_
            if te_best_lambda < te_best:
                # Updating the best test error and weights as a function of the degree of expansion
                te_best = te_best_lambda
                weights_best = weights_best_lambda
                degrees_best[i] = int(j)
                mx_best_tmp = mx
        # Appending the result to the final training data matrix
        mx_best = np.append(mx_best, mx_best_tmp[:, 1:], axis=1)
        print("Done! Best lambda:", best_lambda, ", best degree", degrees_best[i])

    print("Calculating final weights...")
    # Best ridge regression with the obtained polynomial expansion degrees
    for lambda_ in lambdas:
        te_best_final = float('inf')
        te_final, weights_final = k_fold_cross_validation(k, y, mx_best, fun_make_model, lambda_)
        if te_final < te_best_final:
            te_best_final = te_final
            weights_best_final = weights_final
    print("Done!")

    return weights_best_final, degrees_best
