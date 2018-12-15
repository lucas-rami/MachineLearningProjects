# Project Information

## Team Members

* Lucas Ramirez (lucas.ramirez@epfl.ch)
* Manuel Cordova (manuel.cordova@epfl.ch)
* Philippe Weier (philippe.weier@epfl.ch)

## Directory Structure

* **`scripts/` :** Python scripts which create models and train them. Generated models are stored in the **`models/`** directory at the project's root.

* **`src/` :** Contains the API used throughout the project.
  * **`load.py` :** Everything related to loading images from the file system. Data must be located in **`project_files/data`**.
  * **`patch.py` :** Contains functions used to split images into patches as well as functions to reconstruct original images from these patches.
  * **`submission.py` :** Used to transform prediction images (masks) into a submission file with the correct format. All submission files are created in the **`submissions/`** directory at the project's root.

* **`models/` :** Directory where all models (files with `.h5` extension) are stored.

* **`submissions/` :** Directory where all submission files are stored.