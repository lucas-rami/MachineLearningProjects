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
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
import re
import img_manipulation
from importlib import reload
img_manipulation = reload(img_manipulation)
from img_manipulation import *

# Make script reproducible
random.seed(1)

def listdir_nohidden(path):
    list = []
    for f in os.listdir(path):
        if not f.startswith('.'):
            list.append(f)
    return list

nTrain = 200
nAdd = 100

im_height = 200
im_width = 200
gt_height = 200
gt_width = 200
# Loaded a set of images
root_dir = "../../project_files/data/"

training_dir = root_dir + "training/"

image_dir = training_dir + "images/"
files = os.listdir(image_dir)
files.sort()
n = min(nTrain, len(files)) # Load maximum nTrain images
print("Loading " + str(n) + " images...")
imgs = np.asarray([load_image(image_dir + files[i]) for i in range(n)])
print("Done!")
resized_imgs = np.asarray(resize_imgs(imgs,im_height,im_width))
type(imgs[0,0,0,0])

plt.imshow(resized_imgs[50])

gt_dir = training_dir + "groundtruth/"
print("Loading " + str(n) + " images")
gt_imgs = np.asarray([load_image(gt_dir + files[i]) for i in range(n)])
resized_gt_imgs = np.asarray(resize_binary_imgs(gt_imgs,gt_height,gt_width,0.25))
#onehot_gt_imgs = convert_to_one_hot(gt_imgs)
#resized_onehot_gt_imgs = convert_to_one_hot(resized_gt_imgs)
output_gt_imgs = np.expand_dims(resized_gt_imgs, axis=3)
plt.imshow(gt_imgs[50])
plt.imshow(resized_gt_imgs[50])

add_dir = root_dir + "additionalDataset/"
add_image_dir = add_dir + "images/"
add_files = os.listdir(add_image_dir)
add_files.sort()
n2 = min(nAdd, len(add_files)) # Load maximum nTrain images
print("Loading " + str(n2) + " images...")
add_imgs = np.asarray([load_image(add_image_dir + add_files[i]) for i in range(n2)])
print("Done!")
add_img_patches = make_patches(add_imgs,im_height,im_width)
print(add_img_patches.shape)

add_gt_dir = add_dir + "groundtruth/"
print("Loading " + str(n2) + " images")
add_gt_imgs = np.asarray([load_image(add_gt_dir + add_files[i][:-1]) for i in range(n2)])
add_gt_patches = make_patches(add_gt_imgs,im_height,im_width)
print(add_gt_patches.shape)

selection_threshold = 0.7
sel_add_imgs, sel_add_gts = select_imgs(add_img_patches, add_gt_patches, selection_threshold)

#resized_add_gt_patches = np.asarray(resize_binary_imgs(sel_add_gts,gt_height,gt_width,0.25))

output_add_gt_imgs = np.expand_dims(sel_add_gts, axis=3)
print(sel_add_imgs.shape)

test_dir = root_dir + "test_set_images/"
test_files = listdir_nohidden(test_dir)
test_files.sort()
nTest = len(test_files)
print("Loading " + str(nTest) + " images...")
test_imgs = np.asarray([load_image(test_dir + test_files[i]+"/"+test_files[i]+".png") for i in range(nTest)])
print("Done!")
cropped_test_imgs = np.asarray(crop_test_imgs(test_imgs, 400, 400))
print(cropped_test_imgs.shape)
resized_test_imgs = np.asarray(resize_imgs(cropped_test_imgs,im_height,im_width))
print(resized_test_imgs.shape)

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

all_train_imgs = np.append(resized_imgs, sel_add_imgs,axis=0)
all_train_gts = np.append(resized_gt_imgs, sel_add_gts,axis=0)
all_gt_imgs = np.expand_dims(all_train_gts, axis=3)
print(all_train_imgs.shape)
print(all_gt_imgs.shape)
X_train, X_valid, y_train, y_valid = train_test_split(all_train_imgs, all_gt_imgs, test_size=0.15)

epochs = 10
num_classes = 2
alpha = 0.01
batch_size = 32

def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def get_unet(input_img, n_filters=16, dropout=0.5, batchnorm=True):
    # contracting path
    c1 = conv2d_block(input_img, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2)) (c1)
    p1 = Dropout(dropout*0.5)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2)) (c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2)) (c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D(pool_size=(5, 5)) (c4)
    p4 = Dropout(dropout)(p4)

    c5 = conv2d_block(p4, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm)

    # expansive path
    u6 = Conv2DTranspose(n_filters*8, (3, 3), strides=(5, 5), padding='same') (c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model

input_img = Input((im_height, im_width, 3), name='img')
model = get_unet(input_img, n_filters=16, dropout=0.25, batchnorm=True)

model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

callbacks = [EarlyStopping(patience=10, verbose=1),ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),ModelCheckpoint('model-tgs-salt.h5', verbose=1, save_best_only=True, save_weights_only=True)]

model_train = model.fit(X_train, y_train, batch_size=batch_size,epochs=epochs,callbacks=callbacks,verbose=1,validation_data=(X_valid, y_valid))

plt.figure(figsize=(8, 8))
plt.title("Learning curve")
plt.plot(model_train.history["loss"], label="loss")
plt.plot(model_train.history["val_loss"], label="val_loss")
plt.plot( np.argmin(model_train.history["val_loss"]), np.min(model_train.history["val_loss"]), marker="x", color="r", label="best model")
plt.xlabel("Epochs")
plt.ylabel("log_loss")
plt.legend();
plt.savefig('lossplot.png')

model.load_weights('model-tgs-salt.h5')

model.evaluate(all_train_imgs, all_gt_imgs,verbose=1)

model.evaluate(X_valid, y_valid,verbose=1)

predictions_test = model.predict(resized_test_imgs,verbose=1)
predictions_train = model.predict(resized_imgs, verbose=1)

prediction_test_imgs = np.squeeze(merge_prediction_imgs(predictions_test, 304, 304))
prediction_train_imgs = np.squeeze(binarize_imgs(predictions_train, 0.25))
gt_masks = binarize_imgs(prediction_test_imgs, 0.25)

plt.imshow(resized_imgs[25])

plt.imshow(np.squeeze(prediction_train_imgs[25]))

def mask_to_submission_strings(image_filename):
    """Reads a single image and outputs the strings that should go into the submission file"""
    img_number = int(re.search(r"\d+", image_filename).group(0))
    im = mpimg.imread(image_filename)
    patch_size = 16
    for j in range(0, im.shape[1]):
        for i in range(0, im.shape[0]):
            label = im[i, j,0]
            yield("{:03d}_{}_{},{}".format(img_number, j*patch_size, i*patch_size, label))

def masks_to_submission(submission_filename, image_filenames):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for i in range(len(image_filenames)):
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(image_filenames[i]))

prediction_test_imgs = np.squeeze(merge_prediction_imgs(predictions_test, 304, 304))
prediction_train_imgs = np.squeeze(binarize_imgs(predictions_train, 0.5))
gt_masks = binarize_imgs(prediction_test_imgs, 0.25)

plt.imshow(prediction_train_imgs[15])
plt.imshow(resized_gt_imgs[15])

resized_gt_masks = np.asarray(resize_binary_imgs(gt_masks, 38, 38, 0.25))
print(gt_masks.shape)
print(resized_gt_masks.shape)
mask_files = []
for i in range(len(resized_gt_masks)):
    mask_files.append("masks/"+test_files[i]+".tiff")
    im = Image.fromarray(resized_gt_masks[i].astype(np.float32))
    im.save(mask_files[i])



masks_to_submission("test_FCN_200.csv", mask_files)
