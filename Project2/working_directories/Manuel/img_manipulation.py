#-*- coding: utf-8 -*-

# Functions used to manipulate images

import tensorflow as tf
import numpy as np

def crop_input_image(input_img,height,width):
    '''Crop an image into patches of given height and width.'''
    list_imgs = []
    nImgs = np.zeros(2)
    imgwidth = input_img.shape[0]
    imgheight = input_img.shape[1]
    nImgs[0] = imgheight/height
    nImgs[1] = imgwidth/width
    for i in range(int(nImgs[0])):
        for j in range(int(nImgs[1])):
            list_imgs.append(input_img[height*i:height*(i+1),width*j:width*(j+1)])
    return np.asarray(list_imgs)

def make_patches(list_imgs,height,width):
    patches = []
    for i in range(len(list_imgs)):
        patches.append(crop_input_image(list_imgs[i],height,width))
    return np.asarray(patches)

def select_imgs(list_imgs, list_gts, threshold):
    sel_imgs = []
    sel_gts = []
    nH = list_imgs.shape[2]
    nW = list_imgs.shape[3]
    nC = list_imgs.shape[4]
    nMax = threshold*nH*nW*nC
    for i in range(list_imgs.shape[0]):
        for j in range(list_imgs.shape[1]):
            if(sum(sum(sum(list_imgs[i,j]))) < nMax):
                sel_imgs.append(list_imgs[i,j])
                sel_gts.append(list_gts[i,j])
    return np.asarray(sel_imgs), np.asarray(sel_gts)

def crop_4_corners_image(input_img, height, width):
    '''Crop an image into 4 patches of given height and width, each located at
    a corner of the image.'''
    list_imgs = []
    list_imgs.append(input_img[0:height,0:width])
    list_imgs.append(input_img[-(height+1):-1,0:width])
    list_imgs.append(input_img[0:height,-(width+1):-1])
    list_imgs.append(input_img[-(height+1):-1,-(width+1):-1])
    return list_imgs

def crop_test_imgs(input_imgs, height, width):
    '''Cropping of test images done in order to make patches of same size as
    training images.'''
    cropped_imgs = []
    for i in range(len(input_imgs)):
        tmp_imgs = crop_4_corners_image(input_imgs[i],height,width)
        for j in range(len(tmp_imgs)):
            cropped_imgs.append(tmp_imgs[j])
    return cropped_imgs

def merge_test_gt_image(input_imgs, original_height, original_width):
    '''Merging of patches extracted from the cropped test images, used to
    reconstruct the mask corresponding to a whole test image.'''
    merged_img = np.zeros((original_height,original_width,1))
    h = input_imgs[0].shape[0]
    w = input_imgs[0].shape[1]
    merged_img[0:h,0:w] += input_imgs[0]
    merged_img[-(h+1):-1,0:w] += input_imgs[1]
    merged_img[0:h,-(w+1):-1] += input_imgs[2]
    merged_img[-(h+1):-1,-(w+1):-1] += input_imgs[3]
    merged_img[-(h+1):h,:] /= 2
    merged_img[:,-(w+1):w] /= 2
    return merged_img

def convert_to_one_hot(labels):
    '''NOT USED (yet): Convert each entry 0 to [0,1] and each entry 1 to [1,0]
    from groundtruth image.'''
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
    '''Resize images to a given height and width.'''
    with tf.Session() as sess:
        resized_imgs = tf.image.resize_images(input_imgs,(height,width)).eval()
    resized_imgs = resized_imgs
    return resized_imgs

def resize_binary_imgs(input_imgs, height, width, threshold):
    '''Resize groundtruth images to given height and width, with entries 0 or 1
    given by the threshold.'''
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
        bin_imgs.append((img_tmp > threshold).astype(int))
    return bin_imgs


def binarize_imgs(imgs, threshold):
    '''Form mask images from the predictions of the model by quantizing mask
    entries to 0 or 1 according to the threshold set.'''
    bin_imgs = np.empty_like(imgs)
    for i in range(len(imgs)):
        bin_imgs[i] = (imgs[i] > threshold).astype(np.float32)
    return bin_imgs

def merge_prediction_imgs(pred_imgs, original_height, original_width):
    '''Merge together prediction patches obtained by cropping the test images.'''
    i = 0
    pred_gt = []
    while i < len(pred_imgs):
        pred_gt.append(merge_test_gt_image(pred_imgs[i:i+4], original_height, original_width))
        i += 4
    return np.asarray(pred_gt)
