#-*- coding: utf-8 -*-

# Functions used to load training and test images

import os
import numpy as np
from helper import load_image
from img_manipulation import *
import random

def listdir_nohidden(path):
    list = []
    for f in os.listdir(path):
        if not f.startswith('.'):
            list.append(f)
    return list

def load_training(dir, nMax, shuffle=False):
    image_dir = dir + "images/"
    gt_dir = dir + "groundtruth/"
    files = listdir_nohidden(image_dir)
    files.sort()
    n = min(nMax, len(files)) # Load maximum nTrain images
    if shuffle:
        files = np.random.permutation(files)
    print("Loading " + str(n) + " images...")
    imgs = np.asarray([load_image(image_dir + files[i]) for i in range(n)])
    if ".tiff" in files[0]:
        files = [x[:-1] for x in files]
    gt_imgs = np.asarray([load_image(gt_dir + files[i]) for i in range(n)])
    print("Done!")
    return imgs, gt_imgs

def load_test(dir,im_height,im_width):
    test_files = listdir_nohidden(dir)
    test_files.sort()
    nTest = len(test_files)
    print("Loading " + str(nTest) + " images...")
    test_imgs = np.asarray([load_image(dir + test_files[i]+"/"+test_files[i]+".png") for i in range(nTest)])
    print("Done!")
    print("Cropping and resizing images...")
    cropped_test_imgs = np.asarray(crop_test_imgs(test_imgs, 400, 400))
    resized_test_imgs = np.asarray(resize_imgs(cropped_test_imgs,im_height,im_width))
    print("Done!")
    return resized_test_imgs

def training_generator(train_dir, add_dir, nTrain, nAdd, batch_size):
 # Create empty arrays to contain batch of features and labels#
 batch_imgs = np.zeros((batch_size, 200, 200, 3))
 batch_gt_imgs = np.zeros((batch_size, 200, 200, 1))
 files = os.listdir(image_dir)
 files.sort()
 n = min(nTrain, len(files)) # Load maximum nTrain images
 print("Loading " + str(n) + " images...")
 imgs = np.asarray([load_image(image_dir + files[i]) for i in range(n)])
 print("Done!")
 resized_imgs = np.asarray(resize_imgs(imgs,im_height,im_width))
 while True:
   for i in range(batch_size):
     # choose random index in features
     index= random.choice(len(features),1)
     batch_imgs[i] = some_processing(features[index])
     batch_gt_imgs[i] = labels[index]
   yield batch_features, batch_labels
