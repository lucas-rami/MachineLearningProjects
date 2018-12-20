# Project Information

## Team Members

* Lucas Ramirez (lucas.ramirez@epfl.ch)
* Manuel Cordova (manuel.cordova@epfl.ch)
* Philippe Weier (philippe.weier@epfl.ch)

## Installing the Additional Dataset

We leveraged an additional dataset found online (see report for details) to train our models.

To install the additional dataset follow the following link https://drive.google.com/drive/folders/1dG17dy5RVQd-0nBu54FaMCqU5RFnQOIr?usp=sharing and download the `additionalDatasetSel.zip` archive. You then need to extract the archive under `data/` so that the satellite images and groundtruth are stored respectively under `data/additionalDatasetSel/images/` and `data/additionalDatasetSel/groundtruth/`.

## Running our Code

Our main script is located under `scripts/run.py`. It leverages our combined model approach to the road segmentation problem (see report for details). Our submission already contains the trained models for our *Narrow Mapped U-Net* model and *Fully Spatial U-Net* model under `models/output/` so the script doesn't need to re-train the models before generating its predictions (the script can still take a lot of time to run beause it looks for an optimal combination of the 2 models and an optimal threshold to separate the roads from the rest of the image). The predictions are then stored under `submissions/combined_model.csv`.

However, we left the possibility to re-train any of the two basic models when running `scripts/run.py`. In that regard the script takes up to 2 arguments from the command line:

* If its first argument is `train` then the *Narrow Mapped U-Net* model is re-trained.
* Likewise, if its second argument is `train` then the *Fully Spatial U-Net* model is re-trained.
* If arguments are left unspecified then the models are not re-trained.
* If arguments are provided but are different than the `train` string then the models are not re-trained.

Examples:

* `python script/run.py :` generates the predictions without re-training anything.
* `python script/run.py train:` re-trains the *Narrow Mapped U-Net* model.
* `python script/run.py train train:` re-trains both models.
* `python script/run.py anything train:` re-trains the *Fully Spatial U-Net* model.

A detailed description of our other scripts is provided in the following section.

## Directory Structure

* **`data/` :** Contains all the data used for training, validating, and testing.
  * **`test_set_images/` :** Contains the test set.
  * **`training/` :** Contains the provided training data.
  * **`validation/` :** Contains a fraction of the provided training data used for validation.


* **`models/` :**
  * **`definitions/` :** Contains the declarations of our models in Python files.
    * **`unet_120.py` :** *Narrow Mapped U-Net* model.
    * **`unet_200.py` :** *Fully Spatial U-Net* model.
  * **`output/` :** Contains our trained models (files with `.h5` extension).
    * **`unet_patch_120_rot.py` :** Weights for our *Narrow Mapped U-Net* model, resulting from training the model.
    * **`unet_patch_200_rot.py` :** Weights for our *Fully Spatial U-Net* model, resulting from training the model.

* **`scripts/` :** All of our Python scripts (i.e. files that should be executed directly).
  * **`run.py` :** Main script. See utilization in section **Running our Code**.
  * **`unet_patch_120_rot.py` :** Instantiates and trains our *Narrow Mapped U-Net* model. The computed weights are stored under `models/output/unet_patch_120_rot.h5`. The model corresponding to this sript is stored under `models/definitions/unet_120.py`.
  * **`unet_patch_200_rot.py` :** Instantiates and trains our *Fully Spatial U-Net* model. The computed weights are stored under `models/output/unet_patch_200_rot.h5`. The model corresponding to this sript is stored under `models/definitions/unet_200.py`.
  * **`predict_unet_patch_120_rot.py` :** Makes predictions for our *Narrow Mapped U-Net* model. The script assumes that `unet_patch_120_rot.py` was executed before and that the computed weights for this model exist under `models/output/unet_patch_120_rot.h5`. The prediction file for these predictions will be stored under `submissions/unet_patch_120_rot.csv`.
  * **`predict_unet_patch_200_rot.py` :** Makes predictions for our *Fully Spatial U-Net* model. The script assumes that `unet_patch_200_rot.py` was executed before and that the computed weights for this model exist under `models/output/unet_patch_200_rot.h5`. The prediction file for these predictions will be stored under `submissions/unet_patch_200_rot.csv`.
  * **`predict_combined.py` :** Makes predictions for our combined model. The script assumes that both `unet_patch_120_rot.py` and `unet_patch_200_rot.py` were executed before and that the computed weights for these models exist under `models/output/unet_patch_120_rot.h5` and `models/output/unet_patch_200_rot.h5`. The prediction file for these predictions will be stored under `submissions/combined_model.csv`.

* **`src/` :** Contains the API used throughout the project.
  * **`load.py` :** Everything related to loading images from the file system. Data must be located in `project_files/data`.
  * **`patch.py` :** Contains functions used to split images into patches as well as functions to reconstruct original images from these patches.
  * **`submission.py` :** Used to transform prediction images (masks) into a submission file with the correct format. All submission files are created in the `submissions/` directory at the project's root.
  * **`transformation.py` :** Transform images in different ways to augment the dataset.

* **`submissions/` :** Directory where all submission files are stored.