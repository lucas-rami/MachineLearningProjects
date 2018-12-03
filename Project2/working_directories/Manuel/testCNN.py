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

random.seed(1)

def crop_input_image(input_img, height, width):
    list_imgs = []
    nImgs = np.zeros(2)
    imgwidth = input_img.shape[0]
    imgheight = input_img.shape[1]
    nImgs[0] = imgheight/height
    nImgs[1] = imgwidth/width
    for i in range(int(nImgs[0])):
        for j in range(int(nImgs[1])):
            if np.mean(np.mean(np.mean(input_img[height*i:height*(i+1),width*j:width*(j+1)])))!=1:
                list_imgs.append(input_img[height*i:height*(i+1),width*j:width*(j+1)])
    if not list_imgs:
        return
    return list_imgs

def crop_4_corners_image(input_img, height, width):
    list_imgs = []
    list_imgs.append(input_img[0:height,0:width])
    list_imgs.append(input_img[-(height+1):-1,0:width])
    list_imgs.append(input_img[0:height,0:-(width+1):-1])
    list_imgs.append(input_img[-(height+1):-(width+1):-1])
    return list_imgs

def merge_test_gt_image(input_img, height, width):
    list_imgs = []
    nImgs = np.zeros(2)
    imgwidth = input_img.shape[0]
    imgheight = input_img.shape[1]
    nImgs[0] = imgheight/height
    nImgs[1] = imgwidth/width
    for i in range(int(nImgs[0])):
        for j in range(int(nImgs[1])):
            if np.mean(np.mean(np.mean(input_img[height*i:height*(i+1),width*j:width*(j+1)])))!=1:
                list_imgs.append(input_img[height*i:height*(i+1),width*j:width*(j+1)])
    if not list_imgs:
        return
    return list_imgs

def convert_to_one_hot(labels):
    converted_labels = np.zeros((labels.shape[0],labels.shape[1],labels.shape[2],2))
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            for k in range(labels.shape[2]):
                if labels[i,j,k] == 1:
                    converted_labels[i,j,k] = np.asarray([1, 0])
                else:
                    converted_labels[i,j,k] = np.asarray([0, 1])
    return converted_labels

def resize_imgs(input_imgs, height, width):
    with tf.Session() as sess:
        resized_imgs = tf.image.resize_images(input_imgs,(height,width)).eval()
    resized_imgs = resized_imgs
    return resized_imgs

def resize_binary_imgs(input_imgs, height, width):
    bin_imgs = []
    imgwidth = input_imgs[0].shape[0]
    imgheight = input_imgs[0].shape[1]
    nW = int(imgwidth/width)
    nH = int(imgheight/height)
    for i in range(len(input_imgs)):
        img_tmp = np.zeros((height,width))
        for j in range(height):
            for k in range(width):
                img_tmp[j,k] = np.sum(np.sum(input_imgs[i,nH*j:nH*(j+1),nW*k:nW*(k+1)]))
        img_tmp = img_tmp/(nH*nW)
        bin_imgs.append((img_tmp > 0.25).astype(int))
    return bin_imgs

nTrain = 200
nAdd = 100

im_height = 200
im_width = 200
gt_height = 200
gt_width = 200
# Loaded a set of images
root_dir = "../../project_files/data/"

training_dir = root_dir+"training/"

image_dir = training_dir + "images/"
files = os.listdir(image_dir)
files.sort()
n = min(nTrain, len(files)) # Load maximum 20 images
print("Loading " + str(n) + " images...")
imgs = np.asarray([load_image(image_dir + files[i]) for i in range(n)])
print("Done!")
resized_imgs = np.asarray(resize_imgs(imgs,im_height,im_width))
type(imgs[0,0,0,0])

plt.imshow(resized_imgs[50])

gt_dir = training_dir + "groundtruth/"
print("Loading " + str(n) + " images")
gt_imgs = np.asarray([load_image(gt_dir + files[i]) for i in range(n)])
resized_gt_imgs = np.asarray(resize_binary_imgs(gt_imgs,gt_height,gt_width))
#onehot_gt_imgs = convert_to_one_hot(gt_imgs)
resized_onehot_gt_imgs = convert_to_one_hot(resized_gt_imgs)
output_gt_imgs = np.expand_dims(resized_gt_imgs, axis=3)
plt.imshow(gt_imgs[50])
plt.imshow(resized_gt_imgs[50])

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

X_train, X_valid, y_train, y_valid = train_test_split(resized_imgs, output_gt_imgs, test_size=0.15)

epochs = 100
num_classes = 2
alpha = 0.01

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

def get_unet(input_img, n_filters=64, dropout=0.5, batchnorm=True):
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
model = get_unet(input_img, n_filters=16, dropout=0.5, batchnorm=True)

model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

model_train = model.fit(X_train, y_train, batch_size=25,epochs=epochs,verbose=1,validation_data=(X_valid, y_valid))

test_eval = model.evaluate(X_valid, y_valid, verbose=1)

n = 10 # Only use 10 images for training

print('Image size = ' + str(imgs[0].shape[0]) + ',' + str(imgs[0].shape[1]))

# Show first image and its groundtruth image
cimg = concatenate_images(imgs[0], gt_imgs[0])
fig1 = plt.figure(figsize=(10, 10))
plt.imshow(cimg, cmap='Greys_r')

# Extract patches from input images
patch_size = 16 # each patch is 16*16 pixels

img_patches = [img_crop(imgs[i], patch_size, patch_size) for i in range(n)]
gt_patches = [img_crop(gt_imgs[i], patch_size, patch_size) for i in range(n)]

# Linearize list of patches
img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])
gt_patches =  np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])

# Compute features for each image patch
foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch
X = np.asarray([ extract_features_2d(img_patches[i]) for i in range(len(img_patches))])
Y = np.asarray([value_to_class(np.mean(gt_patches[i]), foreground_threshold) for i in range(len(gt_patches))])

# Print feature statistics

print('Computed ' + str(X.shape[0]) + ' features')
print('Feature dimension = ' + str(X.shape[1]))
print('Number of classes = ' + str(np.max(Y)))  #TODO: fix, length(unique(Y))

Y0 = [i for i, j in enumerate(Y) if j == 0]
Y1 = [i for i, j in enumerate(Y) if j == 1]
print('Class 0: ' + str(len(Y0)) + ' samples')
print('Class 1: ' + str(len(Y1)) + ' samples')

# Display a patch that belongs to the foreground class
plt.imshow(gt_patches[Y1[3]], cmap='Greys_r')

# Plot 2d features using groundtruth to color the datapoints
plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)

# train a logistic regression classifier

from sklearn import linear_model

# we create an instance of the classifier and fit the data
logreg = linear_model.LogisticRegression(C=1e5, class_weight="balanced")
logreg.fit(X, Y)

# Predict on the training set
Z = logreg.predict(X)

# Get non-zeros in prediction and grountruth arrays
Zn = np.nonzero(Z)[0]
Yn = np.nonzero(Y)[0]

TPR = len(list(set(Yn) & set(Zn))) / float(len(Z))
print('True positive rate = ' + str(TPR))

# Plot features using predictions to color datapoints
plt.scatter(X[:, 0], X[:, 1], c=Z, edgecolors='k', cmap=plt.cm.Paired)

# Run prediction on the img_idx-th image
img_idx = 8

Xi = extract_img_features(image_dir + files[img_idx], patch_size)
Zi = logreg.predict(Xi)
plt.scatter(Xi[:, 0], Xi[:, 1], c=Zi, edgecolors='k', cmap=plt.cm.Paired)
# Display prediction as an image

w = gt_imgs[img_idx].shape[0]
h = gt_imgs[img_idx].shape[1]
predicted_im = label_to_img(w, h, patch_size, patch_size, Zi)
cimg = concatenate_images(imgs[img_idx], predicted_im)
fig1 = plt.figure(figsize=(10, 10)) # create a figure with the default size
plt.imshow(cimg, cmap='Greys_r')


new_img = make_img_overlay(imgs[img_idx], predicted_im)

plt.imshow(new_img)
