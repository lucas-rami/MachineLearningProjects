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
from transformation import *
from modelCNN_200 import *
from submission import *
from shutil import copyfile
import pickle
import math
import load


# Make script reproducible
random.seed(1)

propTrain = 0.9
propAdd = 0.35
nTrain = 90
nAdd = 100

im_height = 200
im_width = 200
gt_height = 200
gt_width = 200
rot_angle = 45
patch_size = 200

# Loaded a set of images

print("Loading training images...")
root_dir = "../../../project_files/data/"
training_dir = root_dir + "training/"
valid_dir = root_dir + "validation/"
imgs, gt_imgs = load.load_training_data(training_dir)
resized_imgs = resize_imgs(imgs,im_height,im_width)
resized_gt_imgs = resize_binary_imgs(gt_imgs,gt_height,gt_width,0.25)
print("Done!")

print("Rotating training images for data augmentation...")
rot_imgs = rotate_imgs(imgs,rot_angle)
rot_gt_imgs = rotate_imgs(gt_imgs,rot_angle)
resized_rot_imgs = resize_imgs(rot_imgs,im_height,im_width)
resized_rot_gt_imgs = resize_binary_imgs(rot_gt_imgs,gt_height,gt_width,0.25)
all_train_imgs = np.append(resized_imgs, resized_rot_imgs, axis=0)
all_train_gts = np.append(resized_gt_imgs, resized_rot_gt_imgs, axis=0)
all_train_gts = np.expand_dims(all_train_gts,axis=3)
print("Done!")

print("Loading validation images...")
val_imgs, val_gt_imgs = load.load_training_data(valid_dir)
resized_val_imgs = resize_imgs(val_imgs,im_height,im_width)
resized_val_gt_imgs = resize_binary_imgs(val_gt_imgs,gt_height,gt_width,0.25)
print("Done!")

print("Rotating validation images for data augmentation...")
rot_val_imgs = rotate_imgs(val_imgs,rot_angle)
rot_val_gt_imgs = rotate_imgs(val_gt_imgs,rot_angle)
resized_rot_val_imgs = resize_imgs(rot_val_imgs,im_height,im_width)
resized_rot_val_gt_imgs = resize_binary_imgs(rot_val_gt_imgs,gt_height,gt_width,0.25)
all_val_imgs = np.append(resized_val_imgs, resized_rot_val_imgs,axis=0)
all_val_gts = np.append(resized_val_gt_imgs, resized_rot_val_gt_imgs,axis=0)
all_val_gts = np.expand_dims(all_val_gts,axis=3)
print("Done!")

add_dir = root_dir + "additionalDatasetSel/"

epochs = 80
num_classes = 1
alpha = 0.01
batch_size = 100
Dropout = 0

input_img = Input((im_height, im_width, 3), name='img')
model = get_unet(input_img, num_classes, n_filters=16, dropout=Dropout, batchnorm=True)

print("Loading additional training images...")
add_imgs, add_gt_imgs,_,_ = load.load_training_data_and_patch(add_dir, patch_size, random_selection=True, proportion=propAdd)
add_gt_imgs = np.expand_dims(add_gt_imgs,axis=3)
print("Done!")
all_imgs = np.append(all_train_imgs, add_imgs,axis=0)
all_gts = np.append(all_train_gts, add_gt_imgs,axis=0)

all_gts = (all_imgs > 0.5).astype(int)
all_val_gts = (all_val_gts > 0.5).astype(int)

if os.path.isfile('test_CNN_200_rot_dropout'+str(Dropout)+'.h5'):
    model.load_weights('test_CNN_200_rot_dropout'+str(Dropout)+'.h5')
model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=[f1_score])
callbacks = [EarlyStopping(patience=10, verbose=1),ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),ModelCheckpoint('test_CNN_200_rot.h5', verbose=1, save_best_only=True, save_weights_only=True)]
model_train = model.fit(all_imgs, all_gts, batch_size=batch_size,epochs=epochs,callbacks=callbacks,verbose=1,validation_data=(all_val_imgs, all_val_gts))
with open('CNN_200_rot_dropout'+str(Dropout)+'.history', 'wb') as file_pi:
    pickle.dump(model_train.history, file_pi)
print("Training done!")
