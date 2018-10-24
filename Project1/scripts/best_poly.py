# -*- coding: utf-8 -*-
""" Finding best polynomial extension for the input data """

import numpy as np
from implementations import *
from manipulate_data import *
from cross_validation import *

def find_best_poly(k, y, tx, fun_make_model, max_degree, lambdas):
    mx_best = np.ones((len(tx), 1))
    degrees_best = np.ones((len(tx[0]), 1))

    for i in range(tx.shape[1]):
        te_best = float('inf')
        te_best_lambda = float('inf')
        print("Optimizing feature", i+1, "/", tx.shape[1])
        for j in range(max_degree):
            mx = build_poly(tx[:, i], j)
            for lambda_ in lambdas:
                te_tmp, weights_tmp = k_fold_cross_validation(k, y, mx, fun_make_model, lambda_)
                if te_tmp < te_best_lambda:
                    te_best_lambda = te_tmp
                    weights_best_lambda = weights_tmp
                    best_lambda = lambda_
            if te_best_lambda < te_best:
                te_best = te_best_lambda
                weights_best = weights_best_lambda
                degrees_best[i] = int(j)
                mx_best_tmp = mx
        mx_best = np.append(mx_best, mx_best_tmp[:, 1:], axis=1)
        print("Done! Best lambda:", best_lambda)

    print("Calculating final weights...")
    for lambda_ in lambdas:
        te_best_final = float('inf')
        te_final, weights_final = k_fold_cross_validation(k, y, mx_best, fun_make_model, lambda_)
        if te_final < te_best_final:
            te_best_final = te_final
            weights_best_final = weights_final
    print("Done!")

    return weights_best_final, degrees_best
