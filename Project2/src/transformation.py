#-*- coding: utf-8 -*-
"""Applying spatial transformation to images."""

import numpy as np


def img_rotate(image, nb_rot_clockwise):
    """Rotates an image clockwise by either 90°, 180°, or 270°.

    Args:
        image (H x W (x Y) tensor): An image.
        nb_rot_clockwise (int): The clockwise rotation to apply (1=90°, 2=180°, 3=270°).
    Returns:
        H x W (x Y) tensor : The rotated image.
    """
    # Argument checking
    if len(image.shape) < 2 or len(image.shape) > 3:
        raise ValueError("Tensor image must have 2 or 3 dimensions.")
    if nb_rot_clockwise < 1 or nb_rot_clockwise > 3:
        raise ValueError("nb_rot_clockwise must be 1, 2, or 3")
        
    # Get height and width
    height = image.shape[0]
    width = image.shape[1]

    is_rgb = len(image.shape) == 3

    # Create a rotated image
    rotated = np.empty()
    if nb_rot_clockwise == 1 or nb_rot_clockwise == 3:
        if is_rgb:
            rotated = np.zeros( shape=(width, height, image.shape[2]) )
        else:
            rotated = np.zeros( shape=(width, height) )
    else:
        rotated = np.zeros( shape=image.shape )


    # Rotate the image
    if nb_rot_clockwise == 1: # 90° clockwise rotation
        for h in range(height):
            h_rot = height - (h + 1)
            for w in range(width):
                w_rot = w
                if is_rgb:
                    rotated[ w_rot , h_rot , : ] = image[h,w,:]
                else:
                    rotated[ w_rot , h_rot ] = image[h,w]

    if nb_rot_clockwise == 2: # 180° clockwise rotation
        for h in range(height):
            h_rot = height - (h + 1)
            for w in range(width):
                w_rot = width - (w + 1)
                if is_rgb:
                    rotated[ h_rot , w_rot , : ] = image[h,w,:]
                else:
                    rotated[ h_rot , w_rot ] = image[h,w]

    else: # 270° clockwise rotation
        for h in range(height): 
            h_rot = h
            for w in range(width):
                w_rot = width - (w + 1)
                if is_rgb:
                    rotated[ w_rot , h_rot , : ] = image[h,w,:]
                else:
                    rotated[ w_rot , h_rot ] = image[h,w]

    return rotated

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
