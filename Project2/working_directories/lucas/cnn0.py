# Core packages
import os

# Local files
import images as helper

# Numpy
import numpy as np

# Keras
from keras.datasets import fashion_mnist
from keras.utils import to_categorical

# Matplotlib
import matplotlib.pyplot as plt

# ========= FILE PATHS =========
DATA_DIR = "../../project_files/data/"
TEST_IMAGES_DIR = DATA_DIR + "test_set_images/"
TRAINING_IMAGES_DIR = DATA_DIR + "training/images/"
GROUNDTRUTH_DIR = DATA_DIR + "training/groundtruth/"

# ========= CONSTANTS =========
MAX_TRAINING_SAMPLES = 100
NB_EPOCHS = 20

# ========= GET TRAINING IMAGES AND GROUNDTRUTH =========

# Decide on a number of training images
training_files = os.listdir(TRAINING_IMAGES_DIR)
nb_training_images = min(len(training_files), MAX_TRAINING_SAMPLES)
print("Number of training samples: " + str(nb_training_images))

# Load training and groundtruth images 
training_images = helper.extract_data(TRAINING_IMAGES_DIR, nb_training_images)
groundtruth_images = helper.extract_labels(GROUNDTRUTH_DIR, nb_training_images)
print('Training data shape : ', training_images.shape)
print('Groudtruth data shape : ', groundtruth_images.shape)

classes = np.unique(groundtruth_images)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes[0])
print('Output classes : ', classes[1])

# ========= BUILD MODEL =========



########  Tuto datacamp
# fashion_model = Sequential()
# fashion_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(28,28,1),padding='same'))
# fashion_model.add(LeakyReLU(alpha=0.1))
# fashion_model.add(MaxPooling2D((2, 2),padding='same'))
# fashion_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
# fashion_model.add(LeakyReLU(alpha=0.1))
# fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
# fashion_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
# fashion_model.add(LeakyReLU(alpha=0.1))                  
# fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
# fashion_model.add(Flatten())
# fashion_model.add(Dense(128, activation='linear'))
# fashion_model.add(LeakyReLU(alpha=0.1))                  
# fashion_model.add(Dense(num_classes, activation='softmax'))
