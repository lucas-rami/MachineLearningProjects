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
from patch import *
from keras.models import *
from keras.layers import *
from keras.regularizers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
import re
import img_manipulation
from importlib import reload
img_manipulation = reload(img_manipulation)
from img_manipulation import *
import img_load
img_load = reload(img_load)
from img_load import *
from modelCNN_100 import *
from shutil import copyfile
import pickle
import math

# Make script reproducible
random.seed(1)

nTrain = 10
nAdd = 10

patch_size = 100

im_height = 200
im_width = 200
gt_height = 200
gt_width = 200

selection_threshold = 0.7

# Loaded a set of images

print("Loading training images...")
root_dir = "../../../project_files/data/"
training_dir = root_dir + "training/"
valid_dir = root_dir + "validation/"
imgs, gt_imgs = load_training(training_dir, nTrain)
resized_imgs = np.asarray(resize_imgs(imgs,im_height,im_width))
onehot_gt_imgs = convert_to_one_hot(gt_imgs)
resized_onehot_gt_imgs = np.asarray(resize_binary_imgs(onehot_gt_imgs,gt_height,gt_width,0.25))
resized_patches = make_patches(resized_imgs,patch_size,patch_size)
resized_gt_patches = make_patches(resized_onehot_gt_imgs,patch_size,patch_size)
print("Done!")

print("Rotating training images for data augmentation...")
with tf.Session() as sess:
    rot_imgs = tf.contrib.image.rotate(imgs, math.pi / 4, interpolation='BILINEAR').eval()
    onehot_rot_gt_imgs = tf.contrib.image.rotate(onehot_gt_imgs, math.pi / 4, interpolation='BILINEAR').eval()
resized_rot_imgs = np.asarray(resize_imgs(rot_imgs,im_height,im_width))
resized_onehot_rot_gt_imgs = np.asarray(resize_binary_imgs(onehot_rot_gt_imgs,gt_height,gt_width,0.25))
resized_rot_patches = make_patches(resized_rot_imgs,patch_size,patch_size)
resized_rot_gt_patches = make_patches(resized_onehot_rot_gt_imgs,patch_size,patch_size)
all_train_imgs = np.append(resized_patches, resized_rot_patches, axis=0)
all_train_gts = np.append(resized_gt_patches, resized_rot_gt_patches, axis=0)
print("Done!")

print("Loading validation images...")
val_imgs, val_gt_imgs = load_training(valid_dir, nTrain)
resized_val_imgs = np.asarray(resize_imgs(val_imgs,im_height,im_width))
val_patches = make_patches(resized_val_imgs,patch_size,patch_size)
onehot_val_gt_imgs = convert_to_one_hot(val_gt_imgs)
resized_onehot_val_gt_imgs = np.asarray(resize_binary_imgs(onehot_val_gt_imgs,gt_height,gt_width,0.25))
resized_onehot_val_gt_patches = make_patches(resized_onehot_val_gt_imgs,patch_size,patch_size)
print("Done!")

print("Rotating validation images for data augmentation...")
with tf.Session() as sess:
    rot_val_imgs = tf.contrib.image.rotate(val_imgs, math.pi / 4, interpolation='BILINEAR').eval()
    onehot_rot_val_gt_imgs = tf.contrib.image.rotate(onehot_val_gt_imgs, math.pi / 4, interpolation='BILINEAR').eval()
resized_rot_val_imgs = np.asarray(resize_imgs(rot_val_imgs,im_height,im_width))
resized_onehot_rot_val_gt_imgs = np.asarray(resize_binary_imgs(onehot_rot_val_gt_imgs,gt_height,gt_width,0.25))
rot_val_patches = make_patches(resized_rot_val_imgs,patch_size,patch_size)
onehot_rot_val_gt_patches = make_patches(resized_onehot_rot_val_gt_imgs,patch_size,patch_size)
all_val_imgs = np.append(val_patches, rot_val_patches,axis=0)
all_val_gts = np.append(resized_onehot_val_gt_patches, onehot_rot_val_gt_patches,axis=0)
print("Done!")

add_dir = root_dir + "additionalDatasetSel/"

epochs = 50
num_classes = 2
batch_size = 200
num_iters = 1

input_img = Input((patch_size, patch_size, 3), name='img')
model = get_unet(input_img, num_classes, n_filters=16, dropout=0.25, batchnorm=True)
for i in range(num_iters):
    print("Iteration " + str(i+1) + "...")
    print("Loading additional training images...")
    add_imgs, add_gt_imgs = load_training(add_dir, nAdd, shuffle=True)
    add_img_patches = make_patches(add_imgs,patch_size,patch_size)
    add_onehot_gt_imgs = convert_to_one_hot(add_gt_imgs)
    add_onehot_gt_patches = make_patches(add_onehot_gt_imgs,patch_size,patch_size)
    print("Done!")
    all_imgs = np.append(all_train_imgs, add_img_patches,axis=0)
    all_gts = np.append(all_train_gts, add_onehot_gt_patches,axis=0)
    if os.path.isfile('test_CNN_100_oneHot_rot.h5'):
        model.load_weights('test_CNN_100_oneHot_rot.h5')
    model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=[f1_score])
    model.summary()
    callbacks = [EarlyStopping(patience=10, verbose=1),ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),ModelCheckpoint('test_CNN_100_oneHot_rot.h5', verbose=1, save_best_only=True, save_weights_only=True)]
    model_train = model.fit(all_imgs, all_gts, batch_size=batch_size,epochs=epochs,callbacks=callbacks,verbose=1,validation_data=(all_val_imgs, all_val_gts))
    copyfile('test_CNN_100_oneHot_rot.h5', 'test_CNN_100_oneHot_rot_iter'+str(i+1)+'.h5')
    with open('CNN_100_onehot_rot_'+str(i+1)+'.history', 'wb') as file_pi:
        pickle.dump(model_train.history, file_pi)
    print("Iteration " + str(i+1) + " done!")
print("Training done!")
