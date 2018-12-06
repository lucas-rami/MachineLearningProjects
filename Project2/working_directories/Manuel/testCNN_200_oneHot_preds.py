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
from img_manipulation import *
from img_load import *
from modelCNN_200 import *
from submission import *
from shutil import copyfile

# Make script reproducible
random.seed(1)

nTrain = 200
nAdd = 50

im_height = 200
im_width = 200
gt_height = 200
gt_width = 200

selection_threshold = 0.7

# Loaded a set of images

root_dir = "../../project_files/data/"
training_dir = root_dir + "training/"
imgs, gt_imgs = load_training(training_dir, nTrain)
resized_imgs = np.asarray(resize_imgs(imgs,im_height,im_width))

plt.imshow(resized_imgs[13])

onehot_gt_imgs = convert_to_one_hot(gt_imgs)
resized_onehot_gt_imgs = np.asarray(resize_binary_imgs(onehot_gt_imgs,gt_height,gt_width,0.25))
plt.imshow(gt_imgs[13])

add_dir = root_dir + "additionalDataset/"

def train_test_split(X, y, test_size=0.15):
    X_train = []
    y_train = []
    X_valid = []
    y_valid = []
    for i in range(len(X)):
        if random.random() > test_size:
            X_train.append(X[i])
            y_train.append(y[i])
        else:
            X_valid.append(X[i])
            y_valid.append(y[i])
    if not y_valid:
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.15)
    return np.asarray(X_train), np.asarray(X_valid), np.asarray(y_train), np.asarray(y_valid)

epochs = 1
num_classes = 2
alpha = 0.01
batch_size = 32
num_iters = 7

input_img = Input((im_height, im_width, 3), name='img')
model = get_unet(input_img, num_classes, n_filters=16, dropout=0.25, batchnorm=True)

model.load_weights('test_CNN_200.h5')



test_dir = root_dir + "test_set_images/"
resized_test_imgs = load_test(test_dir,im_height,im_width)

predictions_test = model.predict(resized_test_imgs,verbose=1)
print(predictions_test)
ratio=1.06
prediction_gts = predictions_to_masks(predictions_test,ratio)
print(predictions_test[1])
print(prediction_gts[1])
prediction_gts = np.expand_dims(prediction_gts,axis=3)
print(prediction_gts.shape)
prediction_test_imgs = merge_prediction_imgs(prediction_gts, 304, 304)
gt_masks = binarize_imgs(prediction_test_imgs, 0.5)

resized_gt_masks = np.squeeze(np.asarray(resize_binary_imgs(gt_masks, 38, 38, 0.8)))
print(gt_masks.shape)
print(resized_gt_masks.shape)
plt.imshow(resized_test_imgs[36])
plt.imshow(np.squeeze(gt_masks[18]))
plt.imshow(np.squeeze(resized_gt_masks[18]))

mask_files = []
test_files = listdir_nohidden(test_dir)
test_files.sort()
for i in range(len(resized_gt_masks)):
    mask_files.append("masks/"+test_files[i]+".tiff")
    im = Image.fromarray(resized_gt_masks[i].astype(np.float32))
    im.save(mask_files[i])

masks_to_submission("test_FCN_200.csv", mask_files)
