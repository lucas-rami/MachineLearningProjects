# -*- coding: utf-8 -*-
"""a function of ploting figures."""
import numpy as np
import matplotlib.pyplot as plt


def cross_validation_visualization_degree(degrees, score):
    """visualization of the curve of score as a function of the
    degree of polynomial expansion."""
    plt.plot(degrees, score, marker=".", color='r')
    plt.xlabel("Degree of expansion ")
    plt.ylabel("Average prediction score")
    plt.grid(True)
    plt.savefig("cross_validation_degree")

def cross_validation_visualization_lambda(lambdas, score):
    """visualization of the curve of score as a function of
    lambda of ridge regression."""
    plt.semilogx(lambdas, score, marker=".", color='r')
    plt.xlabel("Lambda of ridge regression")
    plt.ylabel("Average prediction score")
    plt.grid(True)
    plt.savefig("cross_validation_lambda")

def cross_validation_visualization_k(k, score):
    """visualization of the curve of score as a function of
    lambda of ridge regression."""
    plt.semilogx(k, score, marker=".", color='r')
    plt.xlabel("Number of folds in cross-validation")
    plt.ylabel("Average prediction score")
    plt.grid(True)
    plt.savefig("cross_validation_k")

def visualization_two_features(y, tx, weights, ind):
    """visualize the raw data as well as the classification result."""
    fig, ax = plt.subplots()
    # plot normalized data
    hits = np.where(y == 1)
    misses = np.where(y == -1)
    ax.scatter(
        tx[hits, ind[0]], tx[hits, ind[1]],
        marker='.', color=[0.06, 0.06, 1], s=10)
    ax.scatter(
        tx[misses, ind[0]], tx[misses, ind[1]],
        marker='*', color=[1, 0.06, 0.06], s=10)
    ax.set_xlabel("DER_deltar_tau_lep")
    ax.set_ylabel("PRI_lep_pt")
    ax.grid()
    # plot normalized data with decision boundary
    """feat1 = np.arange(
        np.min(x1)-1, np.max(x1) + 1, step=0.1)
    feat2 = np.arange(
        np.min(x2)-1, np.max(x2) + 1, step=0.1)
    hx, hy = np.meshgrid(feat1, feat2)
    hxy = (np.c_[hx.reshape(-1), hy.reshape(-1)] - mean_x) / std_x
    x_temp = np.c_[np.ones((hxy.shape[0], 1)), hxy]
    prediction = x_temp.dot(w) > 0.
    prediction = prediction.reshape((weight.shape[0], height.shape[0]))
    fig.contourf(hx, hy, prediction, 1)
    fig.set_xlim([min(x1), max(x1)])
    fig.set_ylim([min(x2), max(x2)])"""
    plt.tight_layout()
    plt.savefig("best_features")
