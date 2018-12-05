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
    imgs = np.asarray([load_image(image_dir + files[i]) for i in range(n)])
    if ".tiff" in files[0]:
        files = [x[:-1] for x in files]
    gt_imgs = np.asarray([load_image(gt_dir + files[i]) for i in range(n)])
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

def training_generator(train_dir, add_dir, nTrain, nAdd, batch_size, ratio=0):
    # Create empty arrays to contain batch of features and labels#
    batch_imgs = np.zeros((batch_size, 200, 200, 3))
    batch_output_gt_imgs = np.zeros((batch_size, 200, 200, 2))
    add_image_dir = add_dir + "images/"
    add_gt_dir = add_dir + "groundtruth/"
    train_image_dir = train_dir + "images/"
    train_files = listdir_nohidden(train_image_dir)
    add_files = listdir_nohidden(add_image_dir)
    if ratio == 0:
        frac_train = nTrain/(nTrain+nAdd)
        nTrainActual = min(min(int(batch_size*frac_train),nTrain),len(train_files))
        nAddActual = batch_size-nTrainActual
    elif ratio == float('inf'):
        nTrainActual = min(min(batch_size,nTrain),len(train_files))
        nAddActual = batch_size-nTrainActual
        if nAddActual < 0:
            error("nTrain should be smaller than batch_size!")
    else:
        nTrainActual = min(min(int(ratio*batch_size),nTrain),len(train_files))
        nAddActual = batch_size-nTrainActual
    while True:
        train_files = np.random.permutation(train_files)
        add_files = np.random.permutation(add_files)
        train_imgs, train_gts = load_training(train_dir, nTrainActual,shuffle=True)
        batch_imgs[:nTrainActual] = np.asarray(resize_imgs(train_imgs, 200, 200))
        batch_onehot_train_gts = convert_to_one_hot(train_gts[:nTrainActual])
        batch_output_gt_imgs[:nTrainActual] = np.asarray(resize_binary_imgs(batch_onehot_train_gts,200,200,0.25))
        index_add = 0
        while index_add < nAddActual:
            img_add_tmp = np.expand_dims(load_image(add_image_dir + add_files[index_add]),axis=0)
            gt_add_tmp = np.expand_dims(load_image(add_gt_dir + add_files[index_add][:-1]),axis=0)
            add_img_patches = make_patches(img_add_tmp,200,200)
            add_gt_patches = make_patches(gt_add_tmp,200,200)
            add_onehot_gt_patches = convert_to_one_hot(add_gt_patches)
            sel_add_imgs, sel_add_onehot_gts = select_imgs(add_img_patches, add_onehot_gt_patches, 0.7)
            nAddSel = len(sel_add_imgs)
            if nAddActual-index_add >= nAddSel:
                batch_imgs[nTrainActual+index_add:nTrainActual+index_add+nAddSel] = sel_add_imgs
                batch_output_gt_imgs[nTrainActual+index_add:nTrainActual+index_add+nAddSel] = sel_add_onehot_gts
            else:
                batch_imgs[nTrainActual+index_add:] = sel_add_imgs[:nAddActual-index_add]
                batch_output_gt_imgs[nTrainActual+index_add:] = sel_add_onehot_gts[:nAddActual-index_add]
            index_add += nAddSel
        yield batch_imgs, batch_output_gt_imgs
