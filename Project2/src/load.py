#-*- coding: utf-8 -*-
"""Loading images from the filesystem. All functions in this file make the assumption
that directories `TRAINING_IMAGES_DIR` and `GROUNDTRUTH_DIR` contain images with the
same dimension, and that when sorting the filenames in alphabetical order we get the
correct one-to-one correspondance between images in the two directories. """

import os
import numpy as np
import matplotlib.image as mpimg
import patch

# ========= FILE PATHS =========
DATA_DIR = "../project_files/data/"
TEST_IMAGES_DIR = DATA_DIR + "test_set_images/"
TRAINING_IMAGES_DIR = DATA_DIR + "training/images/"
GROUNDTRUTH_DIR = DATA_DIR + "training/groundtruth/"


def load_training_set(max_nb_images=-1):
    """Loads the training images as well as their corresponding groundtruth images
    from their respective directories.
    
    The function returns empty arrays if either:
        - an image from the training set could not be loaded
        - an image from the groundtruth set could not be loaded
        - the number of training and groundtruth images is different

    Args:
        max_nb_images (int): The maximum number of data samples to load (-1 by default, i.e. no limit).
    Returns:
        N x H x W x Y tensor: A tensor of N RGB(A) training images.
        N x H x W tensor: A tensor of N black and white groundtruth images corresponding to the training samples.
    """

    # Load images
    training_images = load_data(TRAINING_IMAGES_DIR, max_nb_images)
    groundtruth_images = load_data(GROUNDTRUTH_DIR, max_nb_images)

    # Check tensor sizes
    if len(training_images) == 0 or len(groundtruth_images) == 0 or len(training_images) != len(groundtruth_images):
        return np.empty(), np.empty()

    return training_images, groundtruth_images

def load_training_set_and_patch(patch_size, overlap=0, max_nb_images=-1):
    """Loads the training images as well as their corresponding groundtruth images
    from their respective directories and make patches out of them.
    
    See function `img_patch()` in `pacth.py` for the meaning of arguments `patch_size`, 
    `overlap`.

    The function returns empty arrays if either:
    - an image from the training set could not be loaded
    - an image from the groundtruth set could not be loaded
    - the number of training and groundtruth images is different

    Args:
        patch_size (int): The patch size to use to make patches out of the images.
        overlap (int): The minimum amount of horizontal and vertical overlapping between patches (0 by default).
        max_nb_images (int): The maximum number of data samples to load (-1 by default, i.e. no limit).
    Returns:
        (N*M) x H x W x Y tensor: A tensor of N*M RGB(A) patches of training images.
        (N*M) x H x W tensor: A tensor of N*M black and white patches of groundtruth images corresponding to the training samples.
        int: The number of patches per image (M in the returned tensors).
    """

    # Load images
    training_images, groundtruth_images = load_training_set(max_nb_images)
    if len(training_images) == 0 or len(groundtruth_images) == 0:
        return np.empty(), np.empty()

    # Make patches out of everything
    training_patches, nb_patches_per_image = patch.make_patch_and_flatten(training_images, patch_size, overlap)
    groundtruth_patches, _ = patch.make_patch_and_flatten(groundtruth_images, patch_size, overlap)

    return training_patches, groundtruth_patches, nb_patches_per_image

def load_data(path, max_nb_images=-1):
    """Loads images from the directory pointed to by `path`.
    
    The function returns an empty array if any image from the target directory
    could not be loaded. 
    By default the function loads all images contained in the target directory. 
    However, it my load less if `max_nb_images` is set to some positive integer.

    Args:
        path (string): The directory to load images from.
        max_nb_images (int): The maximum number of images to load (-1 by default, i.e. no limit).
    Returns:
        N x H x W (x Y) tensor: A tensor of images.
    """

    # Determine number of images to load
    training_images = os.listdir(path).sort()
    nb_training_images = len(training_images) if (max_nb_images < 0) else min(len(training_images), max_nb_images)

    # Load the images
    imgs = []
    for i in range(nb_training_images):
        image_filename = path + training_images[i]
        if os.path.isfile(image_filename):
            imgs.append(mpimg.imread(image_filename))
        else:
            print ('Failed to load ' + image_filename + ', aborting.')
            return np.empty()

    return np.asarray(imgs)

