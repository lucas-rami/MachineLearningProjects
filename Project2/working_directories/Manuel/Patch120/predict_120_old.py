#!/usr/bin/env python3

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

# Make script reproducible
random.seed(1)

valid_dir = "../../../project_files/data/training/"

patch_size = 120
overlap = 100
num_classes = 1

test_imgs = load.load_test_set()

test_patches,overlap_image,nPatches = patch.make_patch_and_flatten(test_imgs, patch_size, overlap)

val_patches,val_gt_patches,val_overlap_image,val_nPatches = load.load_training_data_and_patch(valid_dir, patch_size=patch_size, random_selection=False, proportion=1.0, overlap=overlap)
val_gt_patches = np.expand_dims(val_gt_patches,axis=3)
# Loaded a set of images

input_img = Input((patch_size, patch_size, 3), name='img')
model = get_unet_120(input_img, num_classes, n_filters=16, dropout=0.4, batchnorm=True)

model.load_weights('../Dropout/UNET_patch_120_0.4__chkpt.h5')

predictions_test = model.predict(test_patches,verbose=1)

predictions_val = model.predict(val_patches,verbose=1)

predictions = patch.reconstruct_from_flatten(np.squeeze(predictions_test), overlap_image, nPatches, overlap)
val_gt_patches = (val_gt_patches > 0.5).astype(float)

threshold_increment = 0.05
threshold = np.arange(0,round(1+threshold_increment,2),threshold_increment)
best_score = 0
best_thr = 0
score_threshold = []
sess = tf.InteractiveSession()
for thr in threshold:
    predictions_val_bin = (predictions_val > thr).astype(int)
    score = sess.run(f1_score(val_gt_patches,predictions_val_bin))
    score_threshold.append(score)
    if score > best_score:
        best_score = score
        best_thr = thr
sess.close()

score_threshold = np.asarray(score_threshold)
np.savetxt('threshold.out', (threshold, score_threshold))

print("Best score: " + str(best_score)+", with threshold: " + str(best_thr))

predictions_bin = (predictions > best_thr).astype(int)

resize_threshold = 0.2
resized_predictions = tr.resize_binary_imgs(np.expand_dims(predictions_bin,axis=3), 38,38, resize_threshold)

test_dir = "../../../project_files/data/test_set_images/"
mask_files = []
full_mask_files = []
full_test_files = []
for i in range(len(predictions)):
    full_test_files.append("test_"+str(i+1)+".tiff")
for i in range(len(predictions)):
    mask_files.append("masks_old/"+full_test_files[i])
    im = Image.fromarray(np.squeeze(resized_predictions)[i].astype(np.float32))
    im.save(mask_files[i])

for i in range(len(predictions_bin)):
    full_mask_files.append("full_masks_old/"+full_test_files[i])
    im = Image.fromarray(np.squeeze(predictions_bin[i]).astype(np.float32))
    im.save(full_mask_files[i])

sub.predictions_to_submission(predictions_bin, "test_patch_120_old_final.csv",resize_threshold)
