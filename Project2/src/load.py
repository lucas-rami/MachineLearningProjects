#-*- coding: utf-8 -*-
"""Loading images from the filesystem. All functions in this file make the assumption
that directories `PROVIDED_DATA_DIR` and `ADDITIONAL_DATA_DIR` contain images and
groundtruth images with the same dimension, and that when sorting the filenames in
alphabetical order we get the correct one-to-one correspondance between images in the
two sub-directories. """

import os
import numpy as np
import matplotlib.image as mpimg
import patch
from transformation import img_uint8_to_float

# ========= FILE PATHS =========
DATA_DIR = "../../../project_files/data/"

TEST_IMAGES_DIR = DATA_DIR + "test_set_images/"
PROVIDED_DATA_DIR = DATA_DIR + "training/"
ADDITIONAL_DATA_DIR = DATA_DIR + "additionalDatasetSel/"

TRAINING_SUB = "images/"
GROUNDTRUTH_SUB = "groundtruth/"

# ========= CONSTANTS =========
SIZE_TEST_SET = 50

def listdir_nohidden(path):
    """Returns the list of non-hidden files in a directory.
    Hidden files are those whose filename starts with ".".

    Args:
        path (string): Directory where to look for files.
    Returns:
        string list: The list of all non-hidden files in the directory.
    """
    list = []
    for f in os.listdir(path):
        if not f.startswith('.'):
            list.append(f)
    return list

def load_training_data_and_patch(path, patch_size, random_selection=False, proportion=1.0, overlap=0):
    """Loads a (possibly random) set of images and corresponding groundtruths
    from the provided directory located at `path`. Make patches out of these
    images before returning.

    See function `img_patch()` in `pacth.py` for the meaning of arguments `patch_size`,
    and `overlap`.

    The function returns empty arrays if either:
    - an image from the training set could not be loaded
    - an image from the groundtruth set could not be loaded
    - the number of training and groundtruth images is different

    Args:
        path (string): Directory from where to load images and groundtruth images (must end with /).
        patch_size (int): The patch size to use to make patches out of the images.
        overlap (int): The minimum amount of horizontal and vertical overlapping between patches (0 by default).
        random_selection (bool): Indicates whether the function decides on the set of images to load randomly (False by default).
        proportion (float): The proportion (between 0 and 1) of the dataset to load (1 by default).
    Returns:
        (N*M) x H x W x Y tensor: A tensor of N*M RGB(A) patches of training images.
        (N*M) x H x W tensor: A tensor of N*M black and white patches of groundtruth images corresponding to the training samples.
        H x W tensor : An "overlap image" to be used during image recontruction from patches.
        int: The number of patches per image (M in the returned tensors).
    """

    # Load images
    training_images, groundtruth_images = load_training_data(path, random_selection, proportion)
    if len(training_images) == 0 or len(groundtruth_images) == 0:
        return np.empty(), np.empty()

    # Make patches out of everything
    training_patches, overlap_image, nb_patches_per_image = patch.make_patch_and_flatten(training_images, patch_size, overlap)
    groundtruth_patches, _ , _= patch.make_patch_and_flatten(groundtruth_images, patch_size, overlap)

    return np.asarray(training_patches), np.asarray(groundtruth_patches), overlap_image, nb_patches_per_image


def load_training_data(path, random_selection=False, proportion=1.0):
    """Loads a (possibly random) set of images and corresponding groundtruths
    from the provided directory located at `path`.

    By default the function loads all images contained in the provided directory.
    If only `proportion` is specified, then the choice of images is not random and
    the function will return a deterministic set of image/groundtruth pair.

    The function returns empty arrays if any selected image could not be loaded,
    or if the `proportion` argument is invalid.

    Args:
        path (string): Directory from where to load images and groundtruth images (must end with /).
        random_selection (bool): Indicates whether the function decides on the set of images to load randomly (False by default).
        proportion (float): The proportion (between 0 and 1) of the dataset to load (1 by default).
    Returns:
        N x H x W x Y tensor: A tensor of N RGB(A) training images.
        N x H x W tensor: A tensor of N black and white groundtruth images corresponding to the training samples.
    """

    # Check argument
    if proportion < 0.0 or proportion > 1.0:
        return np.empty(), np.empty()

    # Look for images in additional dataset folder
    training_images = listdir_nohidden(path + TRAINING_SUB)
    training_images.sort()
    groundtruth_images = listdir_nohidden(path + GROUNDTRUTH_SUB)
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
        image_filename = path + TRAINING_SUB + training_images[i]
        groundtruth_filename = path + GROUNDTRUTH_SUB + groundtruth_images[i]

        if os.path.isfile(image_filename) and os.path.isfile(groundtruth_filename):
            # Load the image and its corresponding groundtruth
            imgs.append(img_uint8_to_float(mpimg.imread(image_filename)))
            gts.append(img_uint8_to_float(mpimg.imread(groundtruth_filename)))
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
