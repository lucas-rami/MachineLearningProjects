# Project Information

## Team Members

* Lucas Ramirez (lucas.ramirez@epfl.ch)
* Manuel Cordova (manuel.cordova@epfl.ch)
* Philippe Weier (philippe.weier@epfl.ch)

## Running Our Code

## Directory Structure

* **`scripts/` :** All of our Python scripts (i.e. files that should be executed directly).
  * **`run.py` :** Main script. See utilization in section **Running our Code**.
  * **`unet_patch_120_rot.py` :** Instantiates and train our *Narrow Mapped U-Net* model. The computed weights are stored under `models/output/unet_patch_120_rot.h5`. The model corresponding to this sript is stored under `models/definitions/unet_120.py`.
  * **`unet_patch_200_rot.py` :** Instantiates and train our *Fully Spatial U-Net* model. The computed weights are stored under `models/output/unet_patch_200_rot.h5`. The model corresponding to this sript is stored under `models/definitions/unet_200.py`.
  * **`predict_unet_patch_120_rot.py` :** Make predictions for our *Narrow Mapped U-Net* model. The script assumes that `unet_patch_120_rot.py` was executed before and that the computed weights for this model exist under `models/output/unet_patch_120_rot.h5`. The prediction file for these predictions will be stored under `submissions/`
  * **`predict_unet_patch_200_rot.py` :** Make predictions for our *Fully Spatial U-Net* model. The script assumes that `unet_patch_200_rot.py` was executed before and that the computed weights for this model exist under `models/output/unet_patch_200_rot.h5`.
  * **`predict_combined.py` :** Make predictions for our combined model. The script assumes that both `unet_patch_120_rot.py` and `unet_patch_200_rot.py` were executed before and that the computed weights for these models exist under `models/output/unet_patch_120_rot.h5` and `models/output/unet_patch_200_rot.h5`.

* **`src/` :** Contains the API used throughout the project.
  * **`load.py` :** Everything related to loading images from the file system. Data must be located in `project_files/data`.
  * **`patch.py` :** Contains functions used to split images into patches as well as functions to reconstruct original images from these patches.
  * **`submission.py` :** Used to transform prediction images (masks) into a submission file with the correct format. All submission files are created in the `submissions/` directory at the project's root.
  * **`transformation.py` :** Transform images in different ways to augment the dataset.

* **`models/` :**
  * **`definitions/` :** Contains the declarations of our models in Python files.
    * **`unet_120.py` :** *Narrow Mapped U-Net* model.
    * **`unet_200.py` :** *Fully Spatial U-Net* model.
  * **`output/` :** Contains our trained models (files with `.h5` extension).
    * **`unet_patch_120_rot.py` :** Weights for our *Narrow Mapped U-Net* model, resulting from training the model.
    * **`unet_patch_200_rot.py` :** Weights for our *Fully Spatial U-Net* model, resulting from training the model.


* **`submissions/` :** Directory where all submission files are stored.