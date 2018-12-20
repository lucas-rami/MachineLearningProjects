#!/usr/bin/env python3

# Removing warnings
import warnings
warnings.filterwarnings("ignore")

import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.misc as sp
import os,sys
from PIL import Image
from helper import *
import keras
import random
from keras.models import *
from keras.layers import *
from keras.regularizers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
import re
import patch
import load
import transformation as tr
from modelUNET_120 import *
from shutil import copyfile
import math
import submission as sub

def f1_custom(labels, preds):
    true_positives = np.sum(labels*preds)
    try:
        precision = true_positives / np.sum(preds)
        recall = true_positives / np.sum(labels)
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        return 0.0
    return f1

# Make script reproducible
random.seed(1)

valid_dir = "../../../project_files/data/training/"

patch_size = 120
overlap = 100
num_classes = 1

print("Loading test set...")
test_imgs = load.load_test_set()
print("Done!")

print("Making patches from test set...")
test_patches,overlap_image,nPatches = patch.make_patch_and_flatten(test_imgs, patch_size, overlap)
print("Done, " + str(len(test_patches)) + " patches made.")

print("Loading validation set and making patches...")
val_patches,val_gt_patches,val_overlap_image,val_nPatches = load.load_training_data_and_patch(valid_dir, patch_size=patch_size, random_selection=False, proportion=1.0, overlap=overlap)
val_gt_patches = np.expand_dims(val_gt_patches,axis=3)
print("Done, " + str(len(val_patches)) + " patches made.")
# Loaded a set of images

print("Loading model...")
input_img = Input((patch_size, patch_size, 3), name='img')
model = get_unet_120(input_img, num_classes, n_filters=16, dropout=0.4, batchnorm=True)

model.load_weights('UNET_patch_120_rot_chkpt.h5')
print("Done!")

print("Making predictions on test patches...")
predictions_test = model.predict(test_patches,verbose=1)
print("Done!")
print("Making predictions on validation patches...")
predictions_val = model.predict(val_patches,verbose=1)
print("Done!")

print("Reconstructing test predictions...")
predictions = patch.reconstruct_from_flatten(np.squeeze(predictions_test), overlap_image, nPatches, overlap)
print("Done!")
val_gt_patches = (val_gt_patches > 0.5).astype(int)

print("Looking for best threshold...")
threshold_increment = 0.05
threshold = np.arange(0,1,threshold_increment)
best_score = 0
best_thr = 0
score_threshold = []
for thr in threshold:
    predictions_val_bin = (predictions_val > thr).astype(int)
    score = f1_custom(val_gt_patches,predictions_val_bin)
    score_threshold.append(score)
    if score > best_score:
        best_score = score
        best_thr = thr

score_threshold = np.asarray(score_threshold)
np.savetxt('threshold.out', (threshold, score_threshold))

print("Done, best F1 score: " + str(best_score)+", with threshold: " + str(best_thr))

print("Making submission file and masks...")
predictions_bin = (predictions > best_thr).astype(int)

resize_threshold = 0.25
resized_predictions = tr.resize_binary_imgs(np.expand_dims(predictions_bin,axis=3), 38,38, resize_threshold)

test_dir = "../../../project_files/data/test_set_images/"
mask_files = []
full_mask_files = []
full_test_files = []
for i in range(len(predictions)):
    full_test_files.append("test_"+str(i+1)+".tiff")
for i in range(len(predictions)):
    mask_files.append("masks/"+full_test_files[i])
    im = Image.fromarray(np.squeeze(resized_predictions)[i].astype(np.float32))
    im.save(mask_files[i])

for i in range(len(predictions_bin)):
    full_mask_files.append("full_masks/"+full_test_files[i])
    im = Image.fromarray(np.squeeze(predictions_bin[i]).astype(np.float32))
    im.save(full_mask_files[i])

sub.predictions_to_submission(predictions_bin, "test_patch_120_final_0.25.csv",resize_threshold)
print("Done, submission file saved.")
