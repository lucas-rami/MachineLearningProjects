# README : Project Information

## Team Members

* Lucas Ramirez
* Manuel Cordova
* Philippe Weier

## Directory Structure

* **`scripts/` :** Python scripts which create models and train them.

* **`src/` :** Contains the API used throughout the project.
    * **`load.py` :** Everything related to loading images from the file system. 
    * **`patch.py` :** Contains functions used to split images into patches as well as functions to reconstruct original images from these patches.
    * **`submission.py` :** Used to transform prediction images (masks) into a submission file with the correct format. All submission files are created in the **`submissions/`** directory at the project's root.

* **`submissions/` :** Directory where all submission files are written in.