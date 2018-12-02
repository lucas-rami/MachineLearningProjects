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
from keras.models import *
from keras.layers import *
from keras.regularizers import *
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

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
resized_imgs = np.asarray(resize_imgs(imgs,200, 200))
type(imgs[0,0,0,0])

plt.imshow(resized_imgs[50])

gt_dir = training_dir + "groundtruth/"
print("Loading " + str(n) + " images")
gt_imgs = np.asarray([load_image(gt_dir + files[i]) for i in range(n)])
resized_gt_imgs = np.asarray(resize_binary_imgs(gt_imgs, 25,25))
onehot_gt_imgs = convert_to_one_hot(gt_imgs)
resized_onehot_gt_imgs = convert_to_one_hot(resized_gt_imgs)
plt.imshow(gt_imgs[50])
plt.imshow(resized_gt_imgs[50])

batch_size = 25
epochs = 100
num_classes = 2
alpha = 1

model = Sequential()
model.add(Conv2D(64, (3, 3),input_shape=(200,200,3), padding='same'))
model.add(LeakyReLU(alpha))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(LeakyReLU(alpha))
model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(LeakyReLU(alpha))
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(LeakyReLU(alpha))
model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(LeakyReLU(alpha))
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(LeakyReLU(alpha))
model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
model.add(Conv2D(512, (3, 3), padding='same'))
model.add(LeakyReLU(alpha))
model.add(Conv2D(512, (3, 3), padding='same'))
model.add(LeakyReLU(alpha))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(5,5), padding='same'))
model.add(Conv2D(1024, (3, 3), padding='same'))
model.add(LeakyReLU(alpha))
model.add(Conv2D(1024, (3, 3), padding='same'))
model.add(LeakyReLU(alpha))
model.add(Dropout(0.25))
model.add(UpSampling2D(size=(5,5)))
model.add(Conv2D(512, (3, 3), padding='same'))
model.add(LeakyReLU(alpha))
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(LeakyReLU(alpha))
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(LeakyReLU(alpha))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(LeakyReLU(alpha))
model.add(Conv2D(num_classes, (1, 1), padding='same'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

print(model.layers[1])

model.summary()

model_train = model.fit(resized_imgs[:75], resized_onehot_gt_imgs[:75], batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(resized_imgs[75:], resized_onehot_gt_imgs[75:]))

test_eval = model.evaluate(resized_imgs[75:], resized_onehot_gt_imgs[75:], verbose=0)
print(test_eval)
