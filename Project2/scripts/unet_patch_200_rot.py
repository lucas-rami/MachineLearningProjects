#!/usr/bin/env python3

#-*- coding: utf-8 -*-
"""UNET leveraging rotated images and a patch size of 200."""

# API
import sys
sys.path.append("../src")
sys.path.append("../models")
import load
import patch
import submission
import transformation
from definitions.unet_200 import get_unet_200
from score import f1_score

# External librairies
import numpy as np
from keras.layers import Input
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam

# Name of output file (in ../models/output/) where the model's weight will be stored
OUTPUT_NAME = 'unet_patch_200_rot'

# Size of patches to split images
PATCH_SIZE = 200

# Proportion of original dataset used for training
PROP_TRAIN = 0.9

# Proportion of additional dataset used for training
PROP_ADD = 0.35

# Pixel dimensions for training images
IMG_NUM_DIM = 3 

# Angle of rotation for images
ROT_ANGLE = 45

# Height of resized images
RESIZE_HEIGHT = 200

# Width of resized images
RESIZE_WIDTH = 200

# Threshold when resizing groundtruths
THRESHOLD_GT_RESIZE = 0.25

# Fix RNG for reproducibility
np.random.seed(1)

# Loaded a set of images

# ================== LOAD ORIGINAL TRAINING DATA ==================
print("Loading training data...")
root_dir = "../../../project_files/data/"
training_dir = root_dir + "training/"
valid_dir = root_dir + "validation/"

# Load original training dataset
imgs, gt_imgs = load.load_training_data(load.PROVIDED_DATA_DIR)

# Rotate images and groundtruth
rot_imgs = transformation.imgs_rotate(imgs, ROT_ANGLE)
rot_gt_imgs = transformation.imgs_rotate(gt_imgs, ROT_ANGLE)

# Resize everything
resized_imgs = transformation.imgs_resize(imgs, RESIZE_HEIGHT, RESIZE_WIDTH)
resized_rot_imgs = transformation.imgs_resize(rot_imgs, RESIZE_HEIGHT, RESIZE_WIDTH)
resized_gt_imgs = transformation.groundtruth_resize(gt_imgs, RESIZE_HEIGHT, RESIZE_WIDTH, THRESHOLD_GT_RESIZE)
resized_rot_gt_imgs = transformation.groundtruth_resize(rot_gt_imgs, RESIZE_HEIGHT, RESIZE_WIDTH, THRESHOLD_GT_RESIZE)

# Concatenate normal images with rotated images
all_train_imgs = np.append(resized_imgs, resized_rot_imgs, axis=0)
all_train_gts = np.append(resized_gt_imgs, resized_rot_gt_imgs, axis=0)
all_train_gts = np.expand_dims(all_train_gts,axis=3)

# ================== LOAD VALIDATION DATA ==================
print("Loading validation data...")

# Load validation images
val_imgs, val_gt_imgs = load.load_training_data(valid_dir)

# Rotate validation images and groundtruth
rot_val_imgs = transformation.imgs_rotate(val_imgs, ROT_ANGLE)
rot_val_gt_imgs = transformation.imgs_rotate(val_gt_imgs, ROT_ANGLE)

# Resize everything
resized_val_imgs = transformation.imgs_resize(val_imgs, RESIZE_HEIGHT, RESIZE_WIDTH)
resized_rot_val_imgs = transformation.imgs_resize(rot_val_imgs, RESIZE_HEIGHT, RESIZE_WIDTH)
resized_val_gt_imgs = transformation.groundtruth_resize(val_gt_imgs, RESIZE_HEIGHT, RESIZE_WIDTH, THRESHOLD_GT_RESIZE)
resized_rot_val_gt_imgs = transformation.groundtruth_resize(rot_val_gt_imgs, RESIZE_HEIGHT, RESIZE_WIDTH, THRESHOLD_GT_RESIZE)

# Concatenate normal validation images with rotated validation images
all_val_imgs = np.append(resized_val_imgs, resized_rot_val_imgs,axis=0)
all_val_gts = np.append(resized_val_gt_imgs, resized_rot_val_gt_imgs,axis=0)
all_val_gts = np.expand_dims(all_val_gts,axis=3)

# ================== LOAD ADDITIONAL TRAINING DATA ==================
print("Loading additional training data...")

# Load additional training dataset
add_imgs, add_gt_imgs, _, _ = load.load_training_data_and_patch(load.ADDITIONAL_DATA_DIR, PATCH_SIZE, random_selection=True, proportion=PROP_ADD)
add_gt_imgs = np.expand_dims(add_gt_imgs,axis=3)
all_imgs = np.append(all_train_imgs, add_imgs,axis=0)
all_gts = np.append(all_train_gts, add_gt_imgs,axis=0)

# Make sure that groundtruths are filled with only 0's and 1's
all_gts = (all_imgs > 0.5).astype(int)
all_val_gts = (all_val_gts > 0.5).astype(int)

# ================== CREATE MODEL ==================
print("Creating the model...")

# Maximal number of epochs
EPOCHS = 80

# Number of classes on which to build the model
NUM_CLASSES = 1

# Size of a batch for the model
BATCH_SIZE = 100

# Amount of dropout for the model
DROUPOUT = 0.6

# Set up the model
input_img = Input((RESIZE_HEIGHT, RESIZE_WIDTH, IMG_NUM_DIM), name='img')
model = get_unet_200(input_img, NUM_CLASSES, n_filters=16, dropout=DROUPOUT, batchnorm=True)
model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=[f1_score])
model.summary()

# Set up callbacks
callbacks = [   EarlyStopping(patience=10, verbose=1),
                ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
                ModelCheckpoint(submission.MODELS_OUTPUT_DIR + OUTPUT_NAME +'.h5', verbose=1, save_best_only=True, save_weights_only=True)]

# ================== TRAIN THE MODEL ==================
print("Training the model...")
model_train = model.fit(all_imgs, all_gts, batch_size=BATCH_SIZE, epochs=EPOCHS,
                callbacks=callbacks, verbose=1, validation_data=(all_val_imgs, all_val_gts))

# ================== SAVE THE MODEL ==================
print("Training complete. Saving model's history to " + OUTPUT_NAME)
submission.save_training_history(OUTPUT_NAME, model.history)
