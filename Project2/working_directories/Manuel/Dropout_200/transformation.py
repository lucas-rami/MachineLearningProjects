#-*- coding: utf-8 -*-
"""Applying spatial transformation to images."""

import numpy as np
from helper import *
from skimage.transform import resize, rotate

def img_mirror(image, is_horizontal_mirror):
    """Mirros an image horizontally or verically.

    Args:
        image (H x W (x Y) tensor): An image.
        is_horizontal_mirror (bool): Indicates whether to mirror horizontally (True) or vertically (False).
    Returns:
        H x W (x Y) tensor : The mirrored image.
    """

    # Argument checking
    if len(image.shape) < 2 or len(image.shape) > 3:
        raise ValueError("Tensor image must have 2 or 3 dimensions.")

    # Get height and width
    height = image.shape[0]
    width = image.shape[1]

    is_rgb = len(image.shape) == 3

    # Create a mirrored image
    mirrored = np.zeros( shape=image.shape )

    if is_horizontal_mirror: # Horizontal mirroring

        for h in range(height):
            for w in range(width):
                w_mirror = width - (w + 1)
                if is_rgb:
                    mirrored[ h , w_mirror , : ] = image[h,w,:]
                else:
                    mirrored[ h , w_mirror ] = image[h,w]

    else: # Vertical mirroring

        for h in range(height):
            h_mirror = height - (h + 1)
            for w in range(width):
                if is_rgb:
                    mirrored[ h_mirror , w , : ] = image[h,w,:]
                else:
                    mirrored[ h_mirror , w ] = image[h,w]

    return mirrored

def resize_imgs(input_imgs, height, width):
    '''Resize images to a given height and width.'''
    resized_imgs = []
    for i in range(len(input_imgs)):
        resized_imgs.append(resize(input_imgs[i],(height,width)))
    resized_imgs = np.asarray(resized_imgs)
    ndimen = len(resized_imgs.shape)
    if ndimen == 4:
        if not np.issubdtype(type(resized_imgs[0,0,0,0]), np.float32):
            for i in range(len(resized_imgs)):
                resized_imgs[i] = img_uint8_to_float(resized_imgs[i])
    elif ndimen ==3:
        if not np.issubdtype(type(resized_imgs[0,0,0]), np.float32):
            for i in range(len(resized_imgs)):
                resized_imgs[i] = img_uint8_to_float(resized_imgs[i])
    return resized_imgs

def resize_binary_imgs(input_imgs, height, width, threshold):
    '''Resize groundtruth images to given height and width, with entries 0 or 1
    given by the threshold.'''
    resized_imgs = resize_imgs(input_imgs,height,width)
    return (resized_imgs > threshold).astype(np.float32)

def rotate_imgs(imgs, angle):
    '''Rotate images counter-clockwise with a given angle'''
    rot_imgs = []
    for i in range(len(imgs)):
        rot_imgs.append(rotate(imgs[i], angle))
    return np.asarray(rot_imgs)
