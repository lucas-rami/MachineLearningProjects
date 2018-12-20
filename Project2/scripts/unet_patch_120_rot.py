#!/usr/bin/env python3

#-*- coding: utf-8 -*-
"""UNET leveraging rotated images and a patch size of 120."""

# API
import sys
sys.path.append("../src")
sys.path.append("../models")
import load
import patch
import submission
import transformation
from definitions.unet_120 import get_unet_120
from score import f1_score

# External librairies
import numpy as np
from keras.layers import Input
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam


# Name of output file (in ../models/output/) where the model's weight will be stored
OUTPUT_NAME = 'unet_patch_120_rot'

# Size of patches to split images
PATCH_SIZE = 120

# Proportion of original dataset used for training
PROP_TRAIN = 0.9

# Proportion of additional dataset used for training
PROP_ADD = 0.35

# Pixel dimensions for training images
IMG_NUM_DIM = 3 

# Angle of rotation for images
ROT_ANGLE = 45

def main(argv=None):

    # Fix RNG for reproducibility
    np.random.seed(1)

    # ================== LOAD ORIGINAL TRAINING DATA ==================
    print("Loading training data...")

    # Load original training dataset 
    imgs, gt_imgs, _, _ = load.load_training_data_and_patch(load.PROVIDED_DATA_DIR, PATCH_SIZE, proportion=PROP_TRAIN)

    # Rotate images and groundtruth
    rot_imgs = transformation.imgs_rotate(imgs, ROT_ANGLE)
    rot_gt_imgs = transformation.imgs_rotate(gt_imgs, ROT_ANGLE)

    # Concatenate normal images with rotated images
    all_train_imgs = np.append(imgs, rot_imgs, axis=0)
    all_train_gts = np.append(gt_imgs, rot_gt_imgs, axis=0)
    all_train_gts = np.expand_dims(all_train_gts,axis=3)

    # ================== LOAD VALIDATION DATA ==================
    print("Loading validation data...")

    # Load validation images
    val_imgs, val_gt_imgs, _, _ = load.load_training_data_and_patch(load.VALIDATION_DATA_DIR, PATCH_SIZE)

    # Rotate validation images and groundtruth
    rot_val_imgs = transformation.imgs_rotate(val_imgs, ROT_ANGLE)
    rot_val_gt_imgs = transformation.imgs_rotate(val_gt_imgs, ROT_ANGLE)

    # Concatenate normal validation images with rotated validation images
    all_val_imgs = np.append(val_imgs, rot_val_imgs, axis=0)
    all_val_gts = np.append(val_gt_imgs, rot_val_gt_imgs, axis=0)
    all_val_gts = np.expand_dims(all_val_gts, axis=3)

    # ================== LOAD ADDITIONAL TRAINING DATA ==================
    print("Loading additional training data...")

    # Load additional training dataset
    add_imgs, add_gt_imgs,_,_ = load.load_training_data_and_patch(load.ADDITIONAL_DATA_DIR, PATCH_SIZE, proportion=PROP_ADD)
    add_gt_imgs = np.expand_dims(add_gt_imgs,axis=3)
    all_imgs = np.append(all_train_imgs, add_imgs, axis=0)
    all_gts = np.append(all_train_gts, add_gt_imgs, axis=0)

    # Make sure that groundtruths are filled with only 0's and 1's
    all_gts = (all_gts > 0.5).astype(int)
    all_val_gts = (all_val_gts > 0.5).astype(int)

    # ================== CREATE MODEL ==================
    print("Creating the model...")

    # Amount of dropout for the model
    DROPOUT = 0.4

    # Maximal number of epochs
    EPOCHS = 80

    # Size of a batch for the model
    BATCH_SIZE = 100

    # Number of classes on which to build the model
    NUM_CLASSES = 1

    # Set up the model
    input_img = Input((PATCH_SIZE, PATCH_SIZE, IMG_NUM_DIM), name='img')
    model = get_unet_120(input_img, num_classes=NUM_CLASSES, n_filters=16, dropout=DROPOUT, batchnorm=True)
    model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=[f1_score, 'accuracy'])
    model.summary()

    # Set up callbacks
    callbacks = [   EarlyStopping(patience=10, verbose=1),
                    ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
                    ModelCheckpoint(submission.MODELS_OUTPUT_DIR + OUTPUT_NAME +'.h5', verbose=1, save_best_only=True, save_weights_only=True)]

    # ================== TRAIN THE MODEL ==================
    print("Training the model...")
    _ = model.fit(all_imgs, all_gts, batch_size=BATCH_SIZE, epochs=EPOCHS, 
                    callbacks=callbacks, verbose=1, validation_data=(all_val_imgs, all_val_gts))

    # ================== SAVE THE MODEL ==================
    print("Training complete. Saving model's history to " + OUTPUT_NAME)
    submission.save_training_history(OUTPUT_NAME, model.history)

if __name__ == '__main__':
    main()
