#-*- coding: utf-8 -*-
"""Converting predictions to submission format."""

import numpy as np

# Patch size for submission file
PATCH_SIZE_SUBMISSION = 16

# The directory where to store submissions
SUBMISSION_DIR = "../submissions/"

def patch_to_label(patch, foreground_threshold):
    """Outputs whether a patch should be considered being part of the foreground/background. 
    
    Args:
        patch (PATCH_SIZE_SUBMISSION x PATCH_SIZE_SUBMISSION tensor): A prediction image's patch.
        foreground_threshold (float): The average pixel gray value above which a patch is considered to be in the foreground.
    Returns: 
        int: 1 if the patch is considered being part of the foreground, 0 otherwise (part of background).
    """

    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0

def prediction_to_submission_strings(prediction, img_id, foreground_threshold):
    """Yields the strings that should go into the submission file for a single `prediction` image. 
    
    Args:
        prediction (H x W tensor): A prediction image.
        img_id (int): The prediction image ID (starting from 1).
        foreground_threshold (float): The average pixel gray value above which a patch is considered to be in the foreground.
    Returns:
        string generator: The set of strings that should be written into the submission file. 
    """

    for j in range(0, prediction.shape[1], PATCH_SIZE_SUBMISSION):
        for i in range(0, prediction.shape[0], PATCH_SIZE_SUBMISSION):
            patch = prediction[i:i + PATCH_SIZE_SUBMISSION, j:j + PATCH_SIZE_SUBMISSION]
            label = patch_to_label(patch, foreground_threshold)
            yield("{:03d}_{}_{},{}".format(img_id, j, i, label))


def predictions_to_submission(predictions, submission_filename, foreground_threshold):
    """Converts a list of prediction images into a submission file that will be placed into the
    `SUBMISSION_DIR` directory.
    
    Args:
        predictions (N x H x W tensor): A list of prediction images.
        submission_filename (string): The filename for the newly created submission file.
        foreground_threshold (float): The average pixel gray value above which a patch is considered to be in the foreground. 
    """
    
    with open(SUBMISSION_DIR + submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for i in range(len(predictions)):
            f.writelines('{}\n'.format(s) for s in prediction_to_submission_strings(predictions[i], i + 1, foreground_threshold))

