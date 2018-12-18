
# local imports
from patch import *
from img_load import load_training

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

overlap = 0
patch_size = 120
Dropout = 0

nTrain = 90
nValid = 10
nAdd = 1
im_width = 400
im_height = 400
num_dim = 3 # RGB images

#Fix random generation
np.random.seed(1)

root_dir = "../../../project_files/data/"
training_dir = root_dir + "training/"
valid_dir = root_dir + "validation/"
add_dir =root_dir + "additionalDatasetSel/"

imgs, gt_imgs = load_training(training_dir, nTrain)
imgs_val, gt_imgs_val = load_training(valid_dir, nValid)
imgs_add, gt_imgs_add = load_training(add_dir, nAdd)

imgs_patches,overlap_imgs,nPatches = make_patch_and_flatten(imgs,patch_size,overlap)
gts_patches,overlap_gts,nPatches = make_patch_and_flatten(gt_imgs,patch_size,overlap)
gts_patches = np.expand_dims((gts_patches>0.5).astype(int),axis=3)

imgs_patches_val,overlap_imgs_val,nPatches_val = make_patch_and_flatten(imgs_val,patch_size,overlap)
gts_patches_val,overlap_gts_val,nPatches_val = make_patch_and_flatten(gt_imgs_val,patch_size,overlap)
gts_patches_val = np.expand_dims((gts_patches_val>0.5).astype(int),axis=3)

imgs_patches_add,overlap_imgs_add,nPatches_add = make_patch_and_flatten(imgs_add,patch_size,overlap)
gts_patches_add,overlap_gts_add,nPatches_add = make_patch_and_flatten(gt_imgs_add,patch_size,overlap)
gts_patches_add = np.expand_dims((gts_patches_add>0.5).astype(int),axis=3)

all_imgs_train = np.append(imgs_patches, imgs_patches_add, axis=0)
all_gts_train = np.append(gts_patches, gts_patches_add, axis=0)

# We can now get all the datas for training and validation
epochs = 500
batch_size = 100 # important parameter
num_classes = 1

input_img = Input((patch_size, patch_size, num_dim), name='img')

model = get_unet_120(input_img,num_classes=num_classes,n_filters=16, dropout=Dropout, batchnorm=True)

model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=[f1_score, 'accuracy'])
model.summary()

checkpointName = 'UNET_patch_120_' + str(round(Dropout,2)) + '_'
# model.load_weights(checkpointName+'_chkpt.h5')
callbacks = [EarlyStopping(patience=20, verbose=1),
            ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1),
            ModelCheckpoint(checkpointName+'_chkpt.h5', verbose=1, save_best_only=True, save_weights_only=True)]

model_train = model.fit(all_imgs_train, all_gts_train, batch_size=batch_size,epochs=epochs,callbacks=callbacks,verbose=1,validation_data=(imgs_patches_val, gts_patches_val))

copyfile(checkpointName+'_chkpt.h5',checkpointName+'_'+str(epochs)+'epochs.h5')
with open(checkpointName+'.history', 'wb') as file_pi:
    pickle.dump(model_train.history, file_pi)
print("Training done!")
