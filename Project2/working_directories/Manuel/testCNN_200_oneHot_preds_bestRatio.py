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
import math

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
valid_dir = root_dir + "validation/"
imgs, gt_imgs = load_training(valid_dir, nTrain)
resized_imgs = np.asarray(resize_imgs(imgs,im_height,im_width))

plt.imshow(resized_imgs[9])

onehot_gt_imgs = convert_to_one_hot(gt_imgs)
resized_onehot_gt_imgs = np.asarray(resize_binary_imgs(onehot_gt_imgs,gt_height,gt_width,0.25))
plt.imshow(resized_onehot_gt_imgs[9,:,:,0])

num_classes = 2

input_img = Input((im_height, im_width, 3), name='img')
model = get_unet(input_img, num_classes, n_filters=16, dropout=0.25, batchnorm=True)

model.load_weights('test_CNN_200_oneHot_rot_iter3.h5')
model.summary()


test_dir = root_dir + "test_set_images/"
resized_test_imgs = load_test(test_dir,im_height,im_width)

predictions_test = model.predict(resized_test_imgs,verbose=1)
predictions_valid = model.predict(resized_imgs,verbose=1)
ratio=np.arange(0.5,2,0.1)
bestScore = 0
bestRatio = 0
sess = tf.InteractiveSession()
for r in ratio:
    prediction_valid_gts = predictions_to_masks(predictions_valid,r)
    prediction_valid_gts = np.expand_dims(prediction_valid_gts,axis=3)
    gt_valid_masks = binarize_imgs(prediction_valid_gts, 0.25)
    score = sess.run(f1_score(gt_valid_masks[:,:,:,0],resized_onehot_gt_imgs[:,:,:,0]))
    print(score)
    if score > bestScore:
        bestRatio = r
        bestScore = score
    print("Best ratio yet: "+str(bestRatio))
sess.close()
prediction_gts = predictions_to_masks(predictions_test,bestRatio)
prediction_gts = np.expand_dims(prediction_gts,axis=3)
print(prediction_gts.shape)
prediction_test_imgs = merge_prediction_imgs(prediction_gts, 304, 304)
gt_masks = binarize_imgs(prediction_test_imgs, 0.25)
resized_gt_masks = np.squeeze(np.asarray(resize_binary_imgs(gt_masks, 38, 38, 0.01)))
print(gt_masks.shape)
print(resized_gt_masks.shape)
plt.imshow(resized_test_imgs[30])
plt.imshow(np.squeeze(gt_masks[30]))
plt.imshow(np.squeeze(resized_gt_masks[30]))

mask_files = []
full_mask_files = []
test_files = listdir_nohidden(test_dir)
test_files.sort()
for i in range(len(resized_gt_masks)):
    mask_files.append("masks/"+test_files[i]+".tiff")
    im = Image.fromarray(resized_gt_masks[i].astype(np.float32))
    im.save(mask_files[i])


for i in range(len(prediction_test_imgs)):
    full_mask_files.append("full_masks/"+test_files[i]+".tiff")
    im = Image.fromarray(np.squeeze(prediction_test_imgs[i]).astype(np.float32))
    im.save(full_mask_files[i])

masks_to_submission("test_FCN_200_rot_32filters.csv", mask_files)
