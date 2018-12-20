#!/usr/bin/env python3

#-*- coding: utf-8 -*-
"""Combined predictions of our 2 models."""

# API
import sys
sys.path.append("../src")
sys.path.append("../models")
import load
import patch
import submission
import transformation
from definitions.unet_120 import get_unet_120
from definitions.unet_200 import get_unet_200
from score import f1_custom

# External libraries
import numpy as np
from keras.layers import Input

# Name of output file (in ../models/output/) where the model's weight will be stored
OUTPUT_NAME_120 = 'unet_patch_120_rot'
OUTPUT_NAME_200 = 'unet_patch_200_rot'

# Size of patches to split images
PATCH_SIZE_120 = 120
PATCH_SIZE_200 = 200

# Overlapping pixels between patches
OVERLAP_120 = 100
OVERLAP_200 = 190

# Number of classes on which to build the models
NUM_CLASSES = 1

# Pixel dimensions for training images
IMG_NUM_DIM = 3

# Threshold when resizing images
RESIZE_THRESHOLD = 0.25

def main(argv=None):

    # Fix RNG for reproducibility
    np.random.seed(1)

    # Load images
    print("Loading test set...")
    test_imgs = load.load_test_set()

    # ================== unet_patch_12O_rot ==================

    # Make patches out of the testset images
    print("Making patches for model " + OUTPUT_NAME_120)
    test_patches, overlap_test_image, n_test_patches = patch.make_patch_and_flatten(test_imgs, PATCH_SIZE_120, OVERLAP_120)

    # Load validation set (used to determine the best threshold to discriminate foreground from backgound)
    val_imgs,val_gts = load.load_training_data(load.PROVIDED_DATA_DIR)
    # Make sure that groundtruths are filled with only 0's and 1's
    val_gts = (val_gts>0.5).astype(int)

    # Make patches out of the validation images
    val_patches, overlap_val_image, n_val_patches = patch.make_patch_and_flatten(val_imgs, PATCH_SIZE_120, OVERLAP_120)

    # Load first model
    print("Loading model " + OUTPUT_NAME_120)
    input_img_120 = Input((PATCH_SIZE_120, PATCH_SIZE_120, IMG_NUM_DIM), name='img')
    model = get_unet_120(input_img_120, NUM_CLASSES, n_filters=16, dropout=0.4, batchnorm=True)
    model.load_weights(submission.MODELS_OUTPUT_DIR + OUTPUT_NAME_120 + '.h5')

    # Make predictions
    print("Making predictions for model " + OUTPUT_NAME_120)
    predictions_test = model.predict(test_patches, verbose=1)
    predictions_val = model.predict(val_patches, verbose=1)

    # Reconstruct predictions and resize
    predictions_val_120 = patch.reconstruct_from_flatten(np.squeeze(predictions_val), overlap_val_image, n_val_patches, OVERLAP_120)
    predictions_val_120 = transformation.imgs_resize(predictions_val_120, PATCH_SIZE_200, PATCH_SIZE_200)
    predictions_120 = patch.reconstruct_from_flatten(np.squeeze(predictions_test), overlap_test_image, n_test_patches, OVERLAP_120)
    predictions_120 = transformation.imgs_resize(predictions_120, int(test_imgs.shape[1]/2), int(test_imgs.shape[1]/2))

    # ================== unet_patch_200_rot ==================

    # Make patches out of the testset images
    print("Making patches for model " + OUTPUT_NAME_200)
    resized_test_imgs = transformation.imgs_resize(test_imgs, int(test_imgs.shape[1]/2),int(test_imgs.shape[1]/2))
    test_patches, overlap_test_image, n_test_patches = patch.make_patch_and_flatten(resized_test_imgs, PATCH_SIZE_200, OVERLAP_200)

    # Resize images and groundtruths
    resized_val_imgs = transformation.imgs_resize(val_imgs, PATCH_SIZE_200, PATCH_SIZE_200)
    resized_val_gts = transformation.groundtruth_resize(val_gts, PATCH_SIZE_200, PATCH_SIZE_200, RESIZE_THRESHOLD)

    # Load second model
    print("Loading model " + OUTPUT_NAME_200)
    input_img_200 = Input((PATCH_SIZE_200, PATCH_SIZE_200, IMG_NUM_DIM), name='img')
    model = get_unet_200(input_img_200, NUM_CLASSES, n_filters=16, dropout=0.6, batchnorm=True)
    model.load_weights(submission.MODELS_OUTPUT_DIR + OUTPUT_NAME_200 + '.h5')

    # Make predictions
    print("Making predictions for model " + OUTPUT_NAME_200)
    predictions_test = model.predict(test_patches,verbose=1)
    predictions_val_200 = model.predict(resized_val_imgs,verbose=1)

    # Reconstruct predictions
    predictions_val_200 = np.squeeze(predictions_val_200)
    predictions_200 = patch.reconstruct_from_flatten(np.squeeze(predictions_test), overlap_test_image, n_test_patches, OVERLAP_200)

    # ================== FIND BEST THRESHOLD  ==================
    print("Looking for best convex combination of both models...")
    THRESHOLD_INC = 0.05
    RATIO_INC = 0.05
    threshold = np.arange(0.1, round(1+THRESHOLD_INC,2), THRESHOLD_INC)
    ratio = np.arange(0,round(1+RATIO_INC,2), RATIO_INC)
    
    best_overall_score = 0
    for r in ratio:
        
        # Reinitialize the best score and best threshold
        best_score = 0
        best_thr = 0

        # Take convex combination of both predictions
        predictions_val = r * predictions_val_120 + (1 - r) * predictions_val_200
        
        for thr in threshold:
            # Compute predictions on validation set and compute F1 score
            predictions_val_bin = (predictions_val > thr).astype(int)
            score = f1_custom(resized_val_gts,predictions_val_bin)
            
            if score > best_score: # We got our best score yet
                best_score = score
                best_thr = thr

        if best_score > best_overall_score: # We got our best score yet
            best_overall_score = best_score
            best_overall_thr = best_thr
            best_ratio = r

    print("Best score: " + str(round(best_overall_score,4)) + ", with best ratio: " + 
        str(round(best_ratio,2)) +", and best threshold: " + str(round(best_overall_thr,2)))

    # ================== GENERATE SUBMISSION FILE ==================
    COMBINED_OUTPUT = "test_combined_final"
    print("Creating submission file " + COMBINED_OUTPUT)

    predictions = best_ratio * predictions_120 + (1 - best_ratio) * predictions_200
    predictions = (predictions > best_overall_thr).astype(int)

    FOREGROUND_THRESHOLD = 0.25
    submission.predictions_to_submission(predictions, COMBINED_OUTPUT + ".csv", FOREGROUND_THRESHOLD, patch_size_submission=8)

if __name__ == '__main__':
    main()
