#-*- coding: utf-8 -*-
"""Computing the F1 score."""

# External libraires
import tensorflow as tf

def f1_score(y_true, y_pred):
    """Computes the F1 score for a list of predictions `y_pred`.

    Args:
        y_true: List of expected predictions.
        y_pred: List of actual predictions.
    Returns:
        float: The F1 score associated to the predictions.
    """

    y_true = tf.cast(y_true, "int32")
    y_pred = tf.cast(tf.round(y_pred), "int32") # implicit 0.5 threshold via tf.round
    y_correct = y_true * y_pred
    sum_true = tf.reduce_sum(y_true, axis=1)
    sum_pred = tf.reduce_sum(y_pred, axis=1)
    sum_correct = tf.reduce_sum(y_correct, axis=1)
    precision = sum_correct / sum_pred
    recall = sum_correct / sum_true
    f_score = 2 * precision * recall / (precision + recall)
    f_score = tf.where(tf.is_nan(f_score), tf.zeros_like(f_score), f_score)
    return tf.reduce_mean(f_score)