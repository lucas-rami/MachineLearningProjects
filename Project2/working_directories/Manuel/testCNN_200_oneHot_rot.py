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
import img_manipulation
from importlib import reload
img_manipulation = reload(img_manipulation)
from img_manipulation import *
import img_load
img_load = reload(img_load)
from img_load import *
from modelCNN_200 import *
from submission import *
from shutil import copyfile
import pickle
import math

# Make script reproducible
random.seed(1)

nTrain = 90
nAdd = 50

im_height = 200
im_width = 200
gt_height = 200
gt_width = 200

selection_threshold = 0.7

# Loaded a set of images

print("Loading training images...")
root_dir = "../../project_files/data/"
training_dir = root_dir + "training/"
valid_dir = root_dir + "validation/"
imgs, gt_imgs = load_training(training_dir, nTrain)
resized_imgs = np.asarray(resize_imgs(imgs,im_height,im_width))
onehot_gt_imgs = convert_to_one_hot(gt_imgs)
resized_onehot_gt_imgs = np.asarray(resize_binary_imgs(onehot_gt_imgs,gt_height,gt_width,0.25))
print("Done!")

print("Rotating training images for data augmentation...")
with tf.Session() as sess:
    rot_imgs = tf.contrib.image.rotate(imgs, math.pi / 4, interpolation='BILINEAR').eval()
    onehot_rot_gt_imgs = tf.contrib.image.rotate(onehot_gt_imgs, math.pi / 4, interpolation='BILINEAR').eval()
resized_rot_imgs = np.asarray(resize_imgs(rot_imgs,im_height,im_width))
resized_onehot_rot_gt_imgs = np.asarray(resize_binary_imgs(onehot_rot_gt_imgs,gt_height,gt_width,0.25))
all_train_imgs = np.append(resized_imgs, resized_rot_imgs, axis=0)
all_train_gts = np.append(resized_onehot_gt_imgs, resized_onehot_rot_gt_imgs, axis=0)
print("Done!")

print("Loading validation images...")
val_imgs, val_gt_imgs = load_training(valid_dir, nTrain)
resized_val_imgs = np.asarray(resize_imgs(val_imgs,im_height,im_width))
onehot_val_gt_imgs = convert_to_one_hot(val_gt_imgs)
resized_onehot_val_gt_imgs = np.asarray(resize_binary_imgs(onehot_val_gt_imgs,gt_height,gt_width,0.25))
print("Done!")

print("Rotating validation images for data augmentation...")
with tf.Session() as sess:
    rot_val_imgs = tf.contrib.image.rotate(val_imgs, math.pi / 4, interpolation='BILINEAR').eval()
    onehot_rot_val_gt_imgs = tf.contrib.image.rotate(onehot_val_gt_imgs, math.pi / 4, interpolation='BILINEAR').eval()
resized_rot_val_imgs = np.asarray(resize_imgs(rot_val_imgs,im_height,im_width))
resized_onehot_rot_val_gt_imgs = np.asarray(resize_binary_imgs(onehot_rot_val_gt_imgs,gt_height,gt_width,0.25))
all_val_imgs = np.append(resized_val_imgs, resized_rot_val_imgs,axis=0)
all_val_gts = np.append(resized_onehot_val_gt_imgs, resized_onehot_rot_val_gt_imgs,axis=0)
print("Done!")

add_dir = root_dir + "additionalDataset/"

epochs = 20
num_classes = 2
alpha = 0.01
batch_size = 32
num_iters = 3

input_img = Input((im_height, im_width, 3), name='img')
model = get_unet(input_img, num_classes, n_filters=16, dropout=0.25, batchnorm=True)

for i in range(num_iters):
    print("Iteration " + str(i+1) + "...")
    print("Loading additional training images...")
    add_imgs, add_gt_imgs = load_training(add_dir, nAdd, shuffle=True)
    add_img_patches = make_patches(add_imgs,im_height,im_width)
    add_gt_patches = make_patches(add_gt_imgs,im_height,im_width)
    add_onehot_gt_patches = convert_to_one_hot(add_gt_patches)
    sel_add_imgs, sel_add_gts = select_imgs(add_img_patches, add_onehot_gt_patches, selection_threshold)
    print("Done!")
    print("Rotating additional training images for data augmentation...")
    with tf.Session() as sess:
        rot_add_imgs = tf.contrib.image.rotate(sel_add_imgs, math.pi / 4, interpolation='BILINEAR').eval()
        rot_add_gt_imgs = tf.contrib.image.rotate(sel_add_gts, math.pi / 4, interpolation='BILINEAR').eval()
    all_add_imgs = np.append(sel_add_imgs, rot_add_imgs,axis=0)
    all_add_gts = np.append(sel_add_gts, rot_add_gt_imgs,axis=0)
    print("Done!")
    all_imgs = np.append(all_train_imgs, all_add_imgs,axis=0)
    all_gts = np.append(all_train_gts, all_add_gts,axis=0)
    if os.path.isfile('test_CNN_200_oneHot_rot.h5'):
        model.load_weights('test_CNN_200_oneHot_rot.h5')
    model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=[f1_score])
    callbacks = [EarlyStopping(patience=5, verbose=1),ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),ModelCheckpoint('test_CNN_200_oneHot_rot.h5', verbose=1, save_best_only=True, save_weights_only=True)]
    model_train = model.fit(all_imgs, all_gts, batch_size=batch_size,epochs=epochs,callbacks=callbacks,verbose=1,validation_data=(all_val_imgs, all_val_gts))
    copyfile('test_CNN_200_oneHot_iter1.h5', 'test_CNN_200_oneHot_rot_iter'+str(i+1)+'.h5')
    with open('CNN_200_onehot_rot_'+str(i+1)+'.history', 'wb') as file_pi:
        pickle.dump(model_train.history, file_pi)
    print("Iteration " + str(i+1) + " done!")
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
