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

ADDITIONAL_DATA_DIR = DATA_DIR + "additionalDataset/"
ADDITIONAL_TRAINING = ADDITIONAL_DATA_DIR + "images/"
ADDITIONAL_GROUNDTRUTH = ADDITIONAL_DATA_DIR + "groundtruth/"

# ========= CONSTANTS =========
SIZE_TEST_SET = 50

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
    training_images = load_data_from_dir(TRAINING_IMAGES_DIR, max_nb_images)
    groundtruth_images = load_data_from_dir(GROUNDTRUTH_DIR, max_nb_images)

    # Check tensor sizes
    if len(training_images) == 0 or len(groundtruth_images) == 0 or len(training_images) != len(groundtruth_images):
        return np.empty(), np.empty()

    return training_images, groundtruth_images

def load_training_set_and_patch(patch_size, overlap=0, max_nb_images=-1):
    """Loads the training images as well as their corresponding groundtruth images
    from their respective directories and make patches out of them.
    
    See function `img_patch()` in `pacth.py` for the meaning of arguments `patch_size`, 
    and `overlap`.

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
        H x W tensor : An "overlap image" to be used during image recontruction from patches.
        int: The number of patches per image (M in the returned tensors).
    """

    # Load images
    training_images, groundtruth_images = load_training_set(max_nb_images)
    if len(training_images) == 0 or len(groundtruth_images) == 0:
        return np.empty(), np.empty()

    # Make patches out of everything
    training_patches, overlap_image, nb_patches_per_image = patch.make_patch_and_flatten(training_images, patch_size, overlap)
    groundtruth_patches, _ , _= patch.make_patch_and_flatten(groundtruth_images, patch_size, overlap)

    return training_patches, groundtruth_patches, overlap_image, nb_patches_per_image

def load_data_from_dir(path, max_nb_images=-1):
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
    training_images = os.listdir(path)
    training_images.sort()
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

def load_additional_dataset(random_selection=False, proportion=1.0):
    """Loads a (possibly random) set of images and corresponding groundtruths 
    from the additional dataset.
    
    By default the function loads all images contained in the additional dataset.
    If only `proportion` is specified, then the choice of images is not random and
    the function will return a deterministic set of image/groundtruth pair.
      
    The function returns empty arrays if any selected image could not be loaded, 
    or if the `proportion` argument is invalid.

    Args:
        random_selection (bool): Indicates whether the function decides on the set of images to load randomly (False by default).
        proportion (float): The proportion (between 0 and 1) of the additional dataset to load (1 by default).
    Returns:
        N x H x W x Y tensor: A tensor of N RGB(A) training images.
        N x H x W tensor: A tensor of N black and white groundtruth images corresponding to the training samples.
    """

    # Check argument
    if proportion < 0.0 or proportion > 1.0:
        return np.empty(), np.empty()

    # Look for images in additional dataset folder
    training_images = os.listdir(ADDITIONAL_TRAINING)
    training_images.sort()
    groundtruth_images = os.listdir(ADDITIONAL_GROUNDTRUTH)
    groundtruth_images.sort()
    
    # Return immediately if the number of training images and groundtruth is different
    if len(training_images) != len(groundtruth_images):
        return np.empty(), np.empty()

    # Decide on a set of images to select
    indices = np.arange(len(training_images))
    if random_selection:
        indices = np.random.permutation(indices)
    indices = indices[ : int(proportion * len(training_images)) ]

    # Load the images
    imgs = []
    gts = []
    for i in indices:
        image_filename = ADDITIONAL_TRAINING + training_images[i]
        groundtruth_filename = ADDITIONAL_GROUNDTRUTH + groundtruth_images[i]

        if os.path.isfile(image_filename) and os.path.isfile(groundtruth_filename):
            # Load the image and its corresponding groundtruth
            imgs.append(mpimg.imread(image_filename))
            gts.append(mpimg.imread(groundtruth_filename))
        else:
            print ('Failed to load ' + image_filename + ', aborting.')
            return np.empty(), np.empty()

    return np.asarray(imgs), np.asarray(gts)

def load_test_set():
    """Loads all images from the test set.
    
    The function returns an empty array if any image from the test set could 
    not be loaded. 

    Returns:
        SIZE_TEST_SET x H x W x Y tensor: The list of all test images.
    """

    # Get list of filepaths
    test_images = [TEST_IMAGES_DIR + "test_{}/test_{}.png".format(i, i) for i in range(1, SIZE_TEST_SET + 1)]

    # Load the images
    imgs = []
    for image_filename in test_images:
        if os.path.isfile(image_filename):
            imgs.append(mpimg.imread(image_filename))
        else:
            print ('Failed to load ' + image_filename + ', aborting.')
            return np.empty()

    return np.asarray(imgs)




