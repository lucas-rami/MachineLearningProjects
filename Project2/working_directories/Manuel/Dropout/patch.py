#-*- coding: utf-8 -*-
"""Splitting images into (potentially overlapping) patches and reconstructing them."""

import numpy as np

MIN_PATCH_SIZE = 16

def make_patch_and_flatten(images, patch_size, overlap=0):
    """Splits all images from an array into patches, and flattens the resulting array.

    Args:
        images (N x H x W (x Y) tensor): An array of images.
        patch_size (int): The patch size to use to make patches out of the images.
        overlap (int): The minimum amount of horizontal and vertical overlapping between patches (0 by default).
    Returns:
        N*M x patch_size x patch_size (x Y) tensor: The flattened array of patches for all the images.
        H x W tensor : An "overlap image" to be used during image recontruction from patches.
        int: The number of patches per image (M in the returned tensor).
    """

    # Make patches
    patches = [img_patch(images[i], patch_size, overlap)[0] for i in range(len(images))]
    # Flatten the array
    patches_flat = [patches[i][j] for i in range(len(patches)) for j in range(len(patches[i]))]
    # Compute number of patches per image
    nb_patch_per_image = len(patches_flat) / len(images)
    # Get the overlap image
    _, overlap_image = img_patch(images[0], patch_size, overlap)
    # Return
    return np.asarray(patches_flat), overlap_image, nb_patch_per_image

def reconstruct_from_flatten(all_patches, overlap_image, nb_patch_per_image, overlap=0):
    """Reconstructs a set of images from an array of patches belonging to multiple images.

    This function is meant to be used in conjuction with `make_patch_and_flatten()`. It performs its
    inverse operation, i.e. re-constructing the original array of images from the patches and the
    "overlap image" returned by `make_patch_and_flatten()`.

    Example:

        # let images represent an array of images
        patches, overlap_image, nb_patch_per_image  = make_patch_and_flatten(images, patch_size, overlap)
        reconstructed_images = reconstruct_from_flatten(patches, overlap_image, nb_patch_per_image, overlap)
        # we now have images == reconstrcuted_images

    Args:
        patches (N x patch_size x patch_size (x Y) tensor): A list of patches.
        overlap_image (H x W tensor): The "overlap image" associate to the patches.
        overlap (int): The minimum amount of horizontal and vertical overlapping between patches (0 by default).
    Returns:
        H x W (x Y) tensor : The original image reconstructed from the patches.
    """
    reconstructed = []
    nb_images = len(all_patches) / nb_patch_per_image
    index_low = 0
    index_high = nb_patch_per_image

    # Reconstruct all images
    for _ in range(nb_images):
        reconstructed.append(img_reconstruct(all_patches[index_low, index_high], overlap_image, overlap))
        index_low = index_high
        index_high += nb_patch_per_image

    return np.asarray(reconstructed)

def img_patch(image, patch_size, overlap=0):
    """Splits `image` into multiple patches of size `patch_size` * `patch_size`,
    possibly overlapping each other.

    The function returns an "overlap image" which is as big as the original image and in
    which each cell contains the number of patches in which the corresponding pixel in the
    original image can be found. After re-constructing the image from its patches the
    resulting image can be divided by the "overlap image" to obtain the real values for all
    pixels.

    If the `patch_size` is bigger than the image's width or height then it is reduced
    to `min(width,height)` and explicit overlapping is disabled.

    Args:
        image (H x W (x Y) tensor): An image.
        patch_size (int): The patch size to use to make patches out of the image.
        overlap (int): The minimum amount of horizontal and vertical overlapping between patches (0 by default).
    Returns:
        N x patch_size x patch_size (x Y) tensor : The array of patches for the original image.
        H x W tensor : An "overlap image" to be used during image recontruction from patches.
    """

    # Argument checking
    if len(image.shape) < 2 or len(image.shape) > 3:
        raise ValueError("Tensor image must have 2 or 3 dimensions.")
    if patch_size < MIN_PATCH_SIZE:
        raise ValueError("Integer patch_size must be greater than or equal to MIN_PATCH_SIZE.")
    if overlap >= patch_size:
        raise ValueError("overlap must be stricty lesser than patch_size")

    # Get image dimensions
    height = image.shape[0]
    width = image.shape[1]

    # Reduce the patch size and remove overlap if the image is too small
    if patch_size > width or patch_size > height:
        patch_size = min(width, height)
        overlap = 0

    # Compute the size of a patch taking into account overlap values
    patch_overlapped_size = patch_size - overlap

    # Compute number of horizontal/vertical patches
    nb_h_patches = int(height / patch_overlapped_size) - (1 if height % patch_overlapped_size == 0 else 0)
    nb_w_patches = int(width / patch_overlapped_size) - (1 if width % patch_overlapped_size == 0 else 0)

    # Generate patches and overlap image
    overlap_image = np.zeros(shape=(height,width), dtype=int)
    patches = []
    for i in range(nb_h_patches):

        # Determine the vertical bounds
        h_bound_low = i * patch_size
        h_bound_high = (i + 1) * patch_size
        if i == nb_h_patches - 1:
            h_bound_low = height - patch_size
            h_bound_high = height

        for j in range(nb_w_patches):

            # Determine the horizontal bounds
            w_bound_low = j * patch_size
            w_bound_high = (j + 1) * patch_size
            if i == nb_w_patches - 1:
                w_bound_low = width - patch_size
                w_bound_high = width

            # Create new patch
            if len(image.shape) == 2:
                patches.append(image[ h_bound_low:h_bound_high , w_bound_low:w_bound_high ])
            else:
                patches.append(image[ h_bound_low:h_bound_high , w_bound_low:w_bound_high , : ])

            # Iterate over each pixel in the patch to fill the overlap_image
            for k in range(h_bound_low, h_bound_high, 1):
                for l in range(w_bound_low, w_bound_high, 1):
                    overlap_image[l , k] += 1

    return np.asarray(patches), overlap_image

def img_reconstruct(patches, overlap_image, overlap=0):
    """Reconstructs an image from multiple patches, possibly overlapping each other.

    This function is meant to be used in conjuction with `img_patch()`. It performs its
    inverse operation, i.e. re-constructing the original image from the patches and the
    "overlap image" returned by `img_patch()`.

    Example:

        # let image represent an image
        patches, overlap_image = img_patch(image, patch_size, overlap)
        reconstructed_image = img_reconstruct(patches, overlap_image, overlap)
        # we now have image == reconstrcuted_image

    Args:
        patches (N x patch_size x patch_size (x Y) tensor): A list of patches.
        overlap_image (H x W tensor): The "overlap image" associate to the patches.
        overlap (int): The minimum amount of horizontal and vertical overlapping between patches (0 by default).
    Returns:
        H x W (x Y) tensor : The original image reconstructed from the patches.
    """

    # Derive patch size and original width and height from arguments
    patch_size = patches[0].shape[0]
    height = overlap_image.shape[0]
    width = overlap_image.shape[1]

    # Argument checking
    if len(patches[0].shape) < 2 or len(patches[0].shape) > 3:
        raise ValueError("Tensor image must have 2 or 3 dimensions.")
    if overlap_image.shape != (height, width):
        raise ValueError("overlap_image must have shape (width, height).")
    if patch_size < MIN_PATCH_SIZE:
        raise ValueError("Integer patch_size must be greater than or equal to MIN_PATCH_SIZE.")
    if overlap >= patch_size:
        raise ValueError("overlap must be stricty lesser than patch_size")

    # Create array for original image
    image = []
    if len(patches[0].shape) == 2:
        image = np.zeros(shape=( width, height ))
    else:
        image = np.zeros(shape=( width, height, patches[0].shape[2]))

    # Compute the size of a patch taking into account overlap values
    patch_overlapped_size = patch_size - overlap

    # Compute number of horizontal/vertical patches
    nb_h_patches = int(height / patch_overlapped_size) - (1 if height % patch_overlapped_size == 0 else 0)
    nb_w_patches = int(width / patch_overlapped_size) - (1 if width % patch_overlapped_size == 0 else 0)

    for i in range(nb_h_patches):

        # Determine the vertical bounds
        h_bound_low = i * patch_size
        h_bound_high = (i + 1) * patch_size
        if i == nb_h_patches - 1:
            h_bound_low = height - patch_size
            h_bound_high = height

        for j in range(nb_w_patches):

            # Determine the horizontal bounds
            w_bound_low = j * patch_size
            w_bound_high = (j + 1) * patch_size
            if i == nb_w_patches - 1:
                w_bound_low = width - patch_size
                w_bound_high = width

            patch = patches[i * nb_w_patches + j]

            # Iterate over each pixel in the patch to fill the image
            k_patch = 0
            for k in range(h_bound_low, h_bound_high, 1):
                l_patch = 0
                for l in range(w_bound_low, w_bound_high, 1):
                    image[l , k] += patch[k_patch, l_patch]
                    l_patch += 1
                k_patch += 1

    return image / overlap_image
