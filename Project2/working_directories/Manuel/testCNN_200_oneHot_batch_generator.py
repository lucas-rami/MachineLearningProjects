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

# Make script reproducible
random.seed(1)

nTrain = 20
nAdd = 30
nValid = 100

im_height = 200
im_width = 200
gt_height = 200
gt_width = 200

selection_threshold = 0.7

# Loaded a set of images

root_dir = "../../project_files/data/"
train_dir = root_dir + "training/"
add_dir = root_dir + "additionalDataset/"
valid_dir = root_dir + "validation/"

print("Loading validation data...")
imgs_valid, gts_valid = load_training(valid_dir, nValid, shuffle=True)
X_valid = np.asarray(resize_imgs(imgs_valid, 200, 200))
onehot_gts_valid = convert_to_one_hot(gts_valid)
y_valid = np.asarray(resize_binary_imgs(onehot_gts_valid,200,200,0.25))
print("Done!")

print("Initializing model...")
epochs = 5
num_classes = 2
alpha = 0.01
batch_size = 100
input_img = Input((im_height, im_width, 3), name='img')
model = get_unet(input_img, num_classes, n_filters=16, dropout=0.25, batchnorm=True)
model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
model.summary()
callbacks = [EarlyStopping(patience=10, verbose=1),ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),ModelCheckpoint('test_CNN_200_batch.h5', verbose=1, save_best_only=True, save_weights_only=True)]
print("Done!")

print(X_valid.shape)
print(y_valid.shape)
print("Training model...")
model_train = model.fit_generator(training_generator(train_dir, add_dir, nTrain, nAdd, batch_size, ratio=float('inf')), steps_per_epoch=10,epochs=epochs,callbacks=callbacks,verbose=1, validation_data=(X_valid,y_valid))

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
