#-*- coding: utf-8 -*-

# Functions used to manipulate images

import numpy as np
import tensorflow as tf
import keras
from keras.models import *
from keras.layers import *
from keras.regularizers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam

def conv2d_block(input_tensor, n_filters, kernel_size=3,dropout=0.2, batchnorm=False):
    # first layer
    x = Conv2D(filters=n_filters, activation='selu', kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)

    x = Dropout(dropout)(x)
    # second layer
    x = Conv2D(filters=n_filters,activation='selu', kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    return x

def get_unet_2_120(input_img, num_classes = 2, n_filters=16, dropout=0.2, batchnorm=False):
    # contracting path
    c1 = conv2d_block(input_img, n_filters=n_filters*1, kernel_size=3,dropout=dropout, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = conv2d_block(p1, n_filters=n_filters*2, kernel_size=3,dropout=dropout, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = conv2d_block(p2, n_filters=n_filters*4, kernel_size=3,dropout=dropout, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = conv2d_block(p3, n_filters=n_filters*8, kernel_size=3,dropout=dropout, batchnorm=batchnorm)

    # expansive path
    u5 = Conv2DTranspose(n_filters*4, (3, 3),activation='selu', strides=(2, 2), padding='same') (c4)
    u5 = concatenate([u5, c3])
    c5 = conv2d_block(u5, n_filters=n_filters*4, kernel_size=3,dropout=dropout, batchnorm=batchnorm)

    u6 = Conv2DTranspose(n_filters*2, (3, 3),activation='selu', strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c2])
    c6 = conv2d_block(u6, n_filters=n_filters*2, kernel_size=3,dropout=dropout, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters*1, (3, 3),activation='selu', strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c1], axis=3)
    c7 = conv2d_block(u7, n_filters=n_filters*1, kernel_size=3,dropout=dropout, batchnorm=batchnorm)

    outputs = Conv2D(num_classes, (1, 1), activation='sigmoid') (c7)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model

def f1_score(y_true, y_pred):
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
