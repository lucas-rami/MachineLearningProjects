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
from patch import *
from load import *
from modelCNN_200 import *
from submission import *
from shutil import copyfile
import math
from img_load import *
from img_manipulation import *
import submission as sub

# Make script reproducible
random.seed(1)

valid_dir = "../../project_files/data/validation/"

patch_size = 200
overlap = 180
num_classes = 1

test_imgs = load_test_set()
resized_test_imgs = resize_imgs(test_imgs, int(test_imgs[0].shape[0]/2),int(test_imgs[0].shape[1]/2))

test_patches,overlap_image,nPatches = make_patch_and_flatten(resized_test_imgs, patch_size, overlap)
print(test_imgs.shape)
print(test_patches.shape)
print(resized_test_imgs.shape)

val_patches,val_gt_patches,val_overlap_image,val_nPatches = load_training_data_and_patch(valid_dir, patch_size=patch_size*2, random_selection=False, proportion=1.0, overlap=overlap*2)
resized_val_patches = resize_imgs(val_patches,patch_size,patch_size)
val_gt_patches = np.expand_dims(val_gt_patches,axis=3)
resized_val_gt_patches = resize_binary_imgs(val_gt_patches,patch_size,patch_size,0.2)
resized_val_gt_patches = np.squeeze(resized_val_gt_patches)
plt.imshow(resized_val_patches[2])
plt.imshow(resized_val_gt_patches[2])
# Loaded a set of images

plt.imshow(resized_test_imgs[13])

input_img = Input((patch_size, patch_size, 3), name='img')
model = get_unet(input_img, num_classes, n_filters=16, dropout=0.5, batchnorm=True)

model.load_weights('test_CNN_200_rot.h5')
model.summary()

predictions_test = model.predict(test_patches,verbose=1)

predictions_val = model.predict(resized_val_patches,verbose=1)

threshold = np.arange(0.1,1,0.05)

predictions = reconstruct_from_flatten(np.squeeze(predictions_test), overlap_image, nPatches, overlap)


plt.imshow(test_imgs[3])
plt.imshow(predictions[3])

best_score = 0
best_thr = 0

sess = tf.InteractiveSession()
for thr in threshold:
    predictions_val_bin = (predictions_val > thr).astype(int)
    score = sess.run(f1_score(resized_val_gt_patches,np.squeeze(predictions_val_bin)))
    print(score)
    if score > best_score:
        best_score = score
        best_thr = thr
sess.close()

print("Best score: "+ str(best_score)+", with threshold: "+str(best_thr))

predictions_bin = (predictions > best_thr).astype(int)

resize_threshold = 0.25
resized_predictions = resize_binary_imgs(predictions_bin, 38,38, resize_threshold)
plt.imshow(predictions_bin[5])
plt.imshow(resized_predictions[5])


test_dir = "../../../project_files/data/test_set_images/"
mask_files = []
full_mask_files = []
full_test_files = []
for i in range(len(predictions)):
    full_test_files.append("test_"+str(i+1)+".tiff")
for i in range(len(predictions_bin)):
    mask_files.append("masks/"+full_test_files[i])
    im = Image.fromarray(resized_predictions[i].astype(np.float32))
    im.save(mask_files[i])


for i in range(len(predictions_bin)):
    full_mask_files.append("full_masks/"+full_test_files[i])
    im = Image.fromarray(np.squeeze(predictions_bin[i]).astype(np.float32))
    im.save(full_mask_files[i])

sub.masks_to_submission("test_patch_200_rot_thresh_0.25.csv", mask_files)
