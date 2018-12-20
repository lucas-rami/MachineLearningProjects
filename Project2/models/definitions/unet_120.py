#-*- coding: utf-8 -*-
"""UNET for script unet_patch_2OO_rot.py."""

import numpy as np
import keras
from keras.models import Model
from keras.layers import Conv2D, Dropout, MaxPooling2D, Conv2DTranspose, concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam

def conv2d_block(input_tensor, n_filters, kernel_size=3,dropout=0.2, batchnorm=False):
    # First layer
    x = Conv2D(filters=n_filters, activation='selu', kernel_size=(kernel_size, kernel_size), 
        kernel_initializer="he_normal", padding="same")(input_tensor)

    x = Dropout(dropout)(x)

    # Second layer
    x = Conv2D(filters=n_filters,activation='selu', kernel_size=(kernel_size, kernel_size),
        kernel_initializer="he_normal", padding="same")(x)

    return x

def get_unet_120(input_img, num_classes=2, n_filters=16, dropout=0.2, batchnorm=False):
    
    # Contracting path
    c1 = conv2d_block(input_img, n_filters=n_filters*1, kernel_size=3,dropout=dropout, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = conv2d_block(p1, n_filters=n_filters*2, kernel_size=3,dropout=dropout, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = conv2d_block(p2, n_filters=n_filters*4, kernel_size=3,dropout=dropout, batchnorm=batchnorm)
    p3 = MaxPooling2D((6, 6)) (c3)

    c4 = conv2d_block(p3, n_filters=n_filters*8, kernel_size=3,dropout=dropout, batchnorm=batchnorm)
    p4 = MaxPooling2D((5, 5)) (c4)

    c5 = conv2d_block(p4, n_filters=n_filters*16, kernel_size=3,dropout=dropout, batchnorm=batchnorm)

    # Expansive path
    u6 = Conv2DTranspose(n_filters*8, (3, 3),activation='selu', strides=(5, 5), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = conv2d_block(u6, n_filters=n_filters*8, kernel_size=3,dropout=dropout, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters*4, (3, 3),activation='selu', strides=(6, 6), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = conv2d_block(u7, n_filters=n_filters*4, kernel_size=3,dropout=dropout, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters*2, (3, 3),activation='selu', strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = conv2d_block(u8, n_filters=n_filters*2, kernel_size=3,dropout=dropout, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters*1, (3, 3),activation='selu', strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = conv2d_block(u9, n_filters=n_filters*1, kernel_size=3,dropout=dropout, batchnorm=batchnorm)

    outputs = Conv2D(num_classes, (1, 1), activation='sigmoid') (c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model
