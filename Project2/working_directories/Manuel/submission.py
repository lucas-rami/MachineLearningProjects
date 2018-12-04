#-*- coding: utf-8 -*-

# Functions used to create the submission file
import numpy as np
import re
import matplotlib.image as mpimg

def mask_to_submission_strings(image_filename):
    """Reads a single image and outputs the strings that should go into the submission file"""
    img_number = int(re.search(r"\d+", image_filename).group(0))
    im = mpimg.imread(image_filename)
    patch_size = 16
    for j in range(0, im.shape[1]):
        for i in range(0, im.shape[0]):
            label = im[i, j,0]
            yield("{:03d}_{}_{},{}".format(img_number, j*patch_size, i*patch_size, label))

def masks_to_submission(submission_filename, image_filenames):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for i in range(len(image_filenames)):
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(image_filenames[i]))

def predictions_to_masks(preds,ratio):
    masks = []
    for i in range(preds.shape[0]):
        masks.append((preds[i,:,:,0]>preds[i,:,:,1]/ratio).astype(np.uint8))
    return np.asarray(masks)
