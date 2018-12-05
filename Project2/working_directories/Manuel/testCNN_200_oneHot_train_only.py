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
nAdd = 0

im_height = 400
im_width = 400
gt_height = 400
gt_width = 400

selection_threshold = 0.7

# Loaded a set of images

root_dir = "../../project_files/data/"
training_dir = root_dir + "training/"
imgs, gt_imgs = load_training(training_dir, nTrain)

plt.imshow(imgs[13])

onehot_gt_imgs = convert_to_one_hot(gt_imgs)
plt.imshow(gt_imgs[13])
print(onehot_gt_imgs.shape)
print(imgs.shape)

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

epochs = 200
num_classes = 2
alpha = 0.01
batch_size = 32
num_iters = 1

input_img = Input((im_height, im_width, 3), name='img')
model = get_unet(input_img, num_classes, n_filters=16, dropout=0.25, batchnorm=True)

X_train, X_valid, y_train, y_valid = train_test_split(imgs, onehot_gt_imgs, test_size=0.2)
print(X_train.shape)
model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
model.summary()
callbacks = [EarlyStopping(patience=10, verbose=1),ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),ModelCheckpoint('test_CNN_200_train_only.h5', verbose=1, save_best_only=False, save_weights_only=True)]
model_train = model.fit(X_train, y_train, batch_size=batch_size,epochs=epochs,callbacks=callbacks,verbose=1,validation_data=(X_valid, y_valid))
copyfile('test_CNN_200_train_only.h5', 'test_CNN_200_train_only_'+str(i)+'.h5')

print("Training done!")

#test_dir = root_dir + "test_set_images/"
#resized_test_imgs = load_test(test_dir)

#predictions_test = model.predict(resized_test_imgs,verbose=1)


#prediction_test_imgs = np.squeeze(merge_prediction_imgs(predictions_test, 304, 304))
#gt_masks = binarize_imgs(prediction_test_imgs, 0.5)

#resized_gt_masks = np.asarray(resize_binary_imgs(gt_masks, 38, 38, 0.25))
#print(gt_masks.shape)
#print(resized_gt_masks.shape)
#mask_files = []
#for i in range(len(resized_gt_masks)):
#    mask_files.append("masks/"+test_files[i]+".tiff")
#    im = Image.fromarray(resized_gt_masks[i].astype(np.float32))
#    im.save(mask_files[i])

#masks_to_submission("test_FCN_200.csv", mask_files)
