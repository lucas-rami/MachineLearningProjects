
# local imports
import load
import patch
import submission as sub
import transformation as tr

from modelUNET_120 import get_unet_120,f1_score

# librairies imports
import numpy as np
import matplotlib.pyplot as plt
import random
from shutil import copyfile
import pickle
from keras.models import *
from keras.layers import *
from keras.regularizers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam

patch_size = 120
Dropout = 0.4

propTrain = 0.9
propAdd = 0.35
im_width = 400
im_height = 400
num_dim = 3 # RGB images
rot_angle = 45

#Fix random generation
np.random.seed(1)

root_dir = "../../../project_files/data/"
training_dir = root_dir + "training/"
valid_dir = root_dir + "validation/"
add_dir = root_dir + "additionalDatasetSel/"

print("Loading training images...")
imgs, gt_imgs,_,_ = load.load_training_data_and_patch(training_dir, patch_size, proportion=propTrain)
print("Done!")

print("Rotating training images for data augmentation...")
rot_imgs = tr.rotate_imgs(imgs,rot_angle)
rot_gt_imgs = tr.rotate_imgs(gt_imgs,rot_angle)
all_train_imgs = np.append(imgs, rot_imgs, axis=0)
all_train_gts = np.append(gt_imgs, rot_gt_imgs, axis=0)
all_train_gts = np.expand_dims(all_train_gts,axis=3)
print("Done!")

print("Loading validation images...")
val_imgs, val_gt_imgs,_,_ = load.load_training_data_and_patch(valid_dir, patch_size)
print("Done!")

print("Rotating validation images for data augmentation...")
rot_val_imgs = tr.rotate_imgs(val_imgs,rot_angle)
rot_val_gt_imgs = tr.rotate_imgs(val_gt_imgs,rot_angle)
all_val_imgs = np.append(val_imgs, rot_val_imgs,axis=0)
all_val_gts = np.append(val_gt_imgs, rot_val_gt_imgs,axis=0)
all_val_gts = np.expand_dims(all_val_gts,axis=3)
print("Done!")

print("Loading additional images...")
add_imgs, add_gt_imgs,_,_ = load.load_training_data_and_patch(add_dir, patch_size, proportion=propAdd)
add_gt_imgs = np.expand_dims(add_gt_imgs,axis=3)
print("Done!")

all_imgs = np.append(all_train_imgs, add_imgs, axis=0)
all_gts = np.append(all_train_gts, add_gt_imgs, axis=0)
print(all_gts)
# We can now get all the datas for training and validation
epochs = 80
batch_size = 100 # important parameter
num_classes = 1

all_gts = (all_gts > 0.5).astype(int)
all_val_gts = (all_val_gts > 0.5).astype(int)

input_img = Input((patch_size, patch_size, num_dim), name='img')

model = get_unet_120(input_img,num_classes=num_classes,n_filters=16, dropout=Dropout, batchnorm=True)

model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=[f1_score, 'accuracy'])
model.summary()

checkpointName = 'UNET_patch_120_rot'
# model.load_weights(checkpointName+'_chkpt.h5')
callbacks = [EarlyStopping(patience=10, verbose=1),
            ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
            ModelCheckpoint(checkpointName+'_chkpt.h5', verbose=1, save_best_only=True, save_weights_only=True)]

model_train = model.fit(all_imgs, all_gts, batch_size=batch_size,epochs=epochs,callbacks=callbacks,verbose=1,validation_data=(all_val_imgs, all_val_gts))

copyfile(checkpointName+'_chkpt.h5',checkpointName+'_'+str(epochs)+'epochs.h5')
with open(checkpointName+'.history', 'wb') as file_pi:
    pickle.dump(model_train.history, file_pi)
print("Training done!")
