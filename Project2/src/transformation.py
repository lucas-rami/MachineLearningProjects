#-*- coding: utf-8 -*-
"""Applying spatial transformation to images."""

import numpy as np
from skimage.transform import resize, rotate

def img_uint8_to_float(img):
    """Converts an image with pixels in the uint8 format to a float format.
    
    Args:
        img (H x W tensor): An image in the uint8 pixel format
    Returns:
        H x W tensor: the same image with pixels in the float format.
    """
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg)).astype(np.float)
    return rimg

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

def imgs_resize(images, height, width):
    """Resize a list of images to the given `height` and `width`.
    
    Args:
        images (N x H x W (x Y) tensor): A list of images.
        height (int): Height of resized images.
        width (int): Width of resized images.
    Returns:
        (N x height x width (x Y) tensor): A list of resized images.
    """

    # Resize images
    resized_imgs = []
    for i in range(len(images)):
        resized_imgs.append(resize(images[i],(height,width)))
    resized_imgs = np.asarray(resized_imgs)

    # Convert pixels from uint8 to float
    ndimen = len(resized_imgs.shape)
    if ndimen == 4:
        if not np.issubdtype(type(resized_imgs[0,0,0,0]), np.float):
            for i in range(len(resized_imgs)):
                resized_imgs[i] = img_uint8_to_float(resized_imgs[i])
    elif ndimen ==3:
        if not np.issubdtype(type(resized_imgs[0,0,0]), np.float):
            for i in range(len(resized_imgs)):
                resized_imgs[i] = img_uint8_to_float(resized_imgs[i])
    return resized_imgs

def groundtruth_resize(groundtruths, height, width, threshold):
    """Resize groundtruth images to given `height` and `width`, with entries 
    0 or 1 given by the `threshold`.

    Args:
        groundtruths (N x H x W tensor): A list of groundtruth images.
        height (int): Height of resized images.
        width (int): Width of resized images.
        threshold (float): Threshold above which a pixel is considered to be in the foreground (between 0 and 1).
    Returns:
        (N x height x width tensor): A list of resized groundtruths (with pixel values 0 or 1).
    """
    resized_imgs = imgs_resize(groundtruths, height, width)
    return (resized_imgs > threshold).astype(np.float)

def imgs_rotate(imgs, angle):
    """Rotate images counter-clockwise with a given `angle`.
    
    Args:
        imgs (N x H x W (x Y) tensor): A list of images.
        angle (float): The angle of rotation (counter-clockwise).
    Returns:
        N x H x W (x Y) tensor: The list of rotated images.
    """
    rot_imgs = []
    for i in range(len(imgs)):
        rot_imgs.append(rotate(imgs[i], angle))
    return np.asarray(rot_imgs)
