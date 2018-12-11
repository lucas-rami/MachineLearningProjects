#-*- coding: utf-8 -*-
"""Making patches out of images"""

import numpy as np

MIN_PATCH_SIZE = 16

def img_patch(image, patch_size):

    # Argument checking
    if patch_size < MIN_PATCH_SIZE:
        raise ValueError("Integer patch_size must be greater than or equal to MIN_PATCH_SIZE.")
    if len(image.shape) != 3:
        raise ValueError("Tensor image must have 3 dimensions.")

    # Get image dimensions
    width = image.shape[0]
    height = image.shape[1]

    # Reduce the patch size if the image is too small
    if patch_size > width or patch_size > height:
        patch_size = min(width, height)

    width_remaining = width % patch_size
    height_remaining = height % patch_size

    # Get number of horizontal and vertical patches
    horizontal_patches = (width / patch_size) + (1 if width_remaining != 0 else 0)
    vertical_patches = (height / patch_size) + (1 if height_remaining != 0 else 0) 

    # Get number of overlapping pixels horizontally and vertically
    horizontal_overlap = 0 if width_remaining == 0 else (patch_size - width_remaining)
    vertical_overlap = 0 if height_remaining == 0 else (patch_size - height_remaining)

    # Generate overlap image
    overlap_image = np.ndarray(shape=(width,height), dtype=int)
    for i in range(height):
        # Handle vertical overlap
        v_overlap = 0
        if vertical_overlap != 0 and (i >= height - patch_size) and (i < height - height_remaining):
            v_overlap += 1

        for j in range(width):
            # Handle horizontal overlap
            h_overlap = 0
            if horizontal_overlap != 0 and (j >= width - patch_size) and (j < width - width_remaining):
                h_overlap += 1
            overlap_image[i,j] = 1 + h_overlap + v_overlap

    # Generate patches
    patches = []
    for i in range(vertical_patches):
        # Determine the vertical bounds
        v_bound_low = i * patch_size
        v_bound_high = (i + 1) * patch_size
        if vertical_overlap != 0 and i == vertical_patches - 1:
            v_bound_low = height - patch_size
            v_bound_high = height
        
        for j in range(horizontal_patches):
            # Determine the horizontal bounds
            h_bound_low = j * patch_size
            h_bound_high = (j + 1) * patch_size
            if horizontal_overlap != 0 and i == horizontal_patches - 1:
                h_bound_low = width - patch_size
                h_bound_high = width

            # Create new patch
            patches.append(image[ h_bound_low:h_bound_high , v_bound_low:v_bound_high ])
            
    return patches, overlap_image
   


    