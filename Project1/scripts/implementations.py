# -*- coding: utf-8 -*-
"""A function to compute the cost."""
import numpy as np

#************************************************
# Functions needed to implement the required functions
#************************************************

def compute_mse(e):
    """compute the least-squares loss (MSE)"""
    mse = e.dot(e) / (2 * len(e))
    return mse

def compute_gradient_mse(e, tx):
    """Compute the least-squares gradient (MSE)"""
    grad = -1/len(e)*(tx).T.dot(e)
    return grad

def compute_error(y, tx, w):
    """Compute the error vector"""
    return y - tx.dot(w)

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def sigmoid(t):
    """apply sigmoid function on t."""
    # Compute the sigmoid function
    return np.divide(np.exp(t),(1+np.exp(t)))

def calculate_loss_neg_log_likelihood(y, tx, w):
    """compute the cost by negative log likelihood."""
    losses = np.exp(tx.dot(w))
    losses = np.log(1+losses)
    losses = losses-np.multiply(y, tx.dot(w))
    return np.sum(losses)

def calculate_gradient_neg_log_likelihood(y, tx, w):
    """compute the gradient of negative log likelihood loss."""
    a = sigmoid(tx.dot(w))-y
    return tx.T.dot(a)

#************************************************
# Functions to implement
#************************************************

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Performs a linear regression using gradient descent"""
    # Initialize the weights
    w = initial_w
    for _ in range(max_iters):
        # Compute the error
        e = compute_error(y, tx, w)
        # Compute the least squares gradient
        grad_w = compute_gradient_mse(e, tx)
        # Compute the least squares loss
        loss = compute_mse(e)
        # Update the weights by the gradient, scaled by the step size gamma
        w = w-gamma*grad_w
    return w, loss



def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Performs a linear regression using stochastic gradient descent"""
    # Initialize the weights
    w = initial_w
    batch_size = 1
    for batch_y, batch_tx in batch_iter(y, tx, batch_size, num_batches=max_iters, shuffle=True):
        # Compute the error
        batch_e = compute_error(batch_y, batch_tx, w)
        # Compute the stochastic gradient
        stoch_grad_w = compute_gradient_mse(batch_e, batch_tx)
        # Compute the loss
        loss = compute_mse(batch_e)
        # Update the weights by the gradient, scaled by the step size gamma
        w = w-gamma*stoch_grad_w
    return w, loss



def least_squares(y, tx):
    """Performs the least squares regression using the normal equations"""
    # Compute the matrix A
    A = (tx).T.dot(tx)
    # Compute the vector b
    b = (tx).T.dot(y)
    # Compute the solution to the problem Ax = b
    w = np.linalg.solve(A, b)
    # Compute the error vector
    e = compute_error(y, tx, w)
    # Compute the loss
    loss = compute_mse(e)
    return w, loss

def ridge_regression(y, tx, lambda_):
    """Performs a ridge regression using the normal equations"""
    # Compute the vector b
    b = (tx).T.dot(y)
    # Compute the matrix due to the regularizer
    Ridge = 2 * len(y) * lambda_ * np.identity(len(b))
    # Compute the final matrix
    A = (tx).T.dot(tx)+Ridge
    # Compute the solution to the problem Ax = b
    w = np.linalg.solve(A, b)
    # Compute the error vector
    e = compute_error(y, tx, w)
    # Compute the loss
    loss = compute_mse(e)
    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Performs a logistic regression using gradient descent"""
    loss = float('inf')
    w = initial_w
    for iter in range(max_iters):
        # Compute the loss
        newLoss = calculate_loss_neg_log_likelihood(y, tx, w)
        # Compute the gradient
        grad = calculate_gradient_neg_log_likelihood(y, tx, w)
        # Update the weights
        w = w - gamma*grad
        if newLoss-loss < 1e-8:
            return w, newLoss
        loss = newLoss
        print("Iteration", iter)
    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
     """Performs a regularized logistic regression using gradient descent"""
