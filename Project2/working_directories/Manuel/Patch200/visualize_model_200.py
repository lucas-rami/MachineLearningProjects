#!/usr/bin/env python3

# Removing warnings
import warnings
warnings.filterwarnings("ignore")

import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.misc as sp
import os,sys
from PIL import Image
from helper import *
import keras
import random
from keras.models import *
from keras.layers import *
from keras.regularizers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.utils import plot_model
import re
import patch
import load
import transformation as tr
from modelCNN_200 import *
from shutil import copyfile
import math
import submission as sub

# Make script reproducible
random.seed(1)

valid_dir = "../../../project_files/data/training/"

patch_size = 120
overlap = 100
num_classes = 1

print("Loading model...")
input_img = Input((patch_size, patch_size, 3), name='img')
model = get_unet(input_img, num_classes, n_filters=16, dropout=0.4, batchnorm=True)
print("Done!")

plot_model(model, to_file='model_200.png', show_shapes=True, show_layer_names=True)
