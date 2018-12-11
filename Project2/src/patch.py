#-*- coding: utf-8 -*-
"""Splitting images into potentially overlapping patches."""

import numpy as np

MIN_PATCH_SIZE = 16

def img_patch(image, patch_size, h_overlap=0, v_overlap=0):
    """Separates `image` into multiple patches of size `patch_size` * `patch_size`,
    possibly overlapping each other.

    The function returns an "overlap image" which is as big as the original image and in 
    which each cell contains the number of patches in which the corresponding pixel in the
    original image can be found. After re-constructing the image from its patches the 
    resulting image can be divided by the "overlap image" to obtain the real values for all
    pixels. 

    If the `patch_size` is bigger than the image's width or height then it is reduced
    to `min(width,height)` and explicit overlapping is disabled.

    Args:
        image (width * height * X tensor): An image.
        patch_size (int): The patch size to use to make patches out of the image.
        h_overlap (int): The amount of horizontal overlapping between patches.
        v_overlap (int): The amount of vertical overlapping between patches.
    Returns:
        nb_patch * patch_size * patch_size * X tensor : The array of patches for the original image.
        width * height tensor : An "overlap image" to be used during image recontruction from patches.
    """

    # Argument checking
    if len(image.shape) < 2 or len(image.shape) > 3:
        raise ValueError("Tensor image must have 2 or 3 dimensions.")
    if patch_size < MIN_PATCH_SIZE:
        raise ValueError("Integer patch_size must be greater than or equal to MIN_PATCH_SIZE.")
    if h_overlap >= patch_size or v_overlap >= patch_size:
        raise ValueError("h_overlap and v_overlap must be stricty lesser than patch_size") 

    # Get image dimensions
    width = image.shape[0]
    height = image.shape[1]

    # Reduce the patch size and remove overlap if the image is too small
    if patch_size > width or patch_size > height:
        patch_size = min(width, height)
        h_overlap = 0
        v_overlap = 0

    # Compute the size of a patch taking into account overlap values
    h_patch_overlapped_size = patch_size - h_overlap
    v_patch_overlapped_size = patch_size - v_overlap

    # Compute number of horizontal/vertical patches
    nb_h_patches = int(width / h_patch_overlapped_size) - (1 if width % h_patch_overlapped_size == 0 else 0)
    nb_v_patches = int(width / v_patch_overlapped_size) - (1 if height % v_patch_overlapped_size == 0 else 0)

    # Generate patches and overlap image
    overlap_image = np.zeros(shape=(width,height), dtype=int)
    patches = []
    for i in range(nb_v_patches):
        # Determine the vertical bounds
        v_bound_low = i * patch_size
        v_bound_high = (i + 1) * patch_size
        if i == nb_v_patches - 1:
            v_bound_low = height - patch_size
            v_bound_high = height
        
        for j in range(nb_h_patches):
            # Determine the horizontal bounds
            h_bound_low = j * patch_size
            h_bound_high = (j + 1) * patch_size
            if i == nb_h_patches - 1:
                h_bound_low = width - patch_size
                h_bound_high = width

            # Create new patch
            if len(image.shape) == 2:
                patches.append(image[ h_bound_low:h_bound_high , v_bound_low:v_bound_high ])
            else:
                patches.append(image[ h_bound_low:h_bound_high , v_bound_low:v_bound_high , : ])

            # Iterate over each pixel in the patch to fill the overlap_image
            for k in range(v_bound_low, v_bound_high, 1):
                for l in range(h_bound_low, h_bound_high, 1):
                    overlap_image[l , k] += 1

    return np.asarray(patches), overlap_image
    