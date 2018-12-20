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
from modelUNET_120 import *
from modelCNN_200 import get_unet
import submission as sub
import transformation as tr

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

patch_size_1 = 120
patch_size_2 = 200
overlap_1 = 100
overlap_2 = 180
num_classes = 1
resize_threshold = 0.25

test_imgs = load.load_test_set()

# First model
test_patches,overlap_test_image,n_test_patches = patch.make_patch_and_flatten(test_imgs, patch_size_1, overlap_1)

val_imgs,val_gts = load.load_training_data(valid_dir)
val_gts = (val_gts>0.5).astype(int)
val_patches,overlap_val_image,n_val_patches = patch.make_patch_and_flatten(val_imgs, patch_size_1, overlap_1)

input_img_1 = Input((patch_size_1, patch_size_1, 3), name='img')
model = get_unet_120(input_img_1, num_classes, n_filters=16, dropout=0.4, batchnorm=True)

model.load_weights('UNET_patch_120_rot_80epochs.h5')

predictions_test = model.predict(test_patches,verbose=1)

predictions_val = model.predict(val_patches,verbose=1)

predictions_val_120 = patch.reconstruct_from_flatten(np.squeeze(predictions_val), overlap_val_image, n_val_patches, overlap_1)
predictions_val_120 = tr.resize_imgs(predictions_val_120, patch_size_2, patch_size_2)

predictions_120 = patch.reconstruct_from_flatten(np.squeeze(predictions_test), overlap_test_image, n_test_patches, overlap_1)
predictions_120 = tr.resize_imgs(predictions_120, int(test_imgs.shape[1]/2), int(test_imgs.shape[1]/2))

# Second model
resized_test_imgs = tr.resize_imgs(test_imgs, int(test_imgs.shape[1]/2),int(test_imgs.shape[1]/2))
test_patches,overlap_test_image,n_test_patches = patch.make_patch_and_flatten(resized_test_imgs, patch_size_2, overlap_2)

resized_val_imgs = tr.resize_imgs(val_imgs, patch_size_2, patch_size_2)
resized_val_gts = tr.resize_binary_imgs(val_gts, patch_size_2, patch_size_2, resize_threshold)

input_img_2 = Input((patch_size_2, patch_size_2, 3), name='img')
model = get_unet(input_img_2, num_classes, n_filters=16, dropout=0.4, batchnorm=True)

model.load_weights('test_CNN_200_rot.h5')

predictions_test = model.predict(test_patches,verbose=1)

predictions_val_200 = model.predict(resized_val_imgs,verbose=1)
predictions_val_200 = np.squeeze(predictions_val_200)
predictions_200 = patch.reconstruct_from_flatten(np.squeeze(predictions_test), overlap_test_image, n_test_patches, overlap_2)

threshold_increment = 0.05
ratio_increment = 0.05
threshold = np.arange(0.1,round(1+threshold_increment,2),threshold_increment)
ratio = np.arange(0,round(1+ratio_increment,2),ratio_increment)
best_overall_score = 0
ratio_scores = []
sess = tf.InteractiveSession()
for r in ratio:
    best_score = 0
    best_thr = 0
    predictions_val = r*predictions_val_120 + (1-r)*predictions_val_200
    for thr in threshold:
        predictions_val_bin = (predictions_val > thr).astype(int)
        score = sess.run(f1_score(resized_val_gts,predictions_val_bin))
        if score > best_score:
            best_score = score
            best_thr = thr
        if thr > best_thr + 0.2:
            break
    ratio_scores.append(best_score)
    if best_score > best_overall_score:
        best_overall_score = best_score
        best_overall_thr = best_thr
        best_ratio = r
sess.close()

print("Best ratio: " + str(best_ratio) +", best threshold: " + str(best_overall_thr))

predictions = best_ratio * predictions_120 + (1-best_ratio)*predictions_200
predictions = (predictions > best_overall_thr).astype(int)

resized_predictions = tr.resize_binary_imgs(np.expand_dims(predictions,axis=3), 38,38, resize_threshold)
plt.imshow(predictions[2])
plt.imshow(resized_predictions[2,:,:,0])


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

for i in range(len(predictions)):
    full_mask_files.append("full_masks/"+full_test_files[i])
    im = Image.fromarray(np.squeeze(predictions[i]).astype(np.float32))
    im.save(full_mask_files[i])

sub.predictions_to_submission(predictions, "test_combined_final.csv",resize_threshold, patch_size_submission=8)
