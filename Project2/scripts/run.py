#!/usr/bin/env python3

#-*- coding: utf-8 -*-
"""Our submission UNET."""

# Library
import sys

# Other scripts
import unet_patch_120_rot
import unet_patch_200_rot
import predict_combined

# Check the number of arguments
if len(sys.argv > 3):
    print("Too many arguments. See README.md.")
    sys.exit()

# Decide which scripts to run
run_train_120 = False if len(sys.argv) < 2 else (True if sys.argv[1] == "train" else False)
run_train_200 = False if len(sys.argv) < 3 else (True if sys.argv[2] == "train" else False)

# Run training if requested
if run_train_120:
    unet_patch_120_rot.main()
if run_train_200:
    unet_patch_200_rot.main()

# Run predictions
predict_combined.main()
