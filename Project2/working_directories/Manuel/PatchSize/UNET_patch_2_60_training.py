
# local imports
from patch import img_patch
from img_load import load_training
from img_manipulation import convert_to_one_hot

from modelUNET_2_60 import get_unet_2_60,f1_score

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

PATCH_SIZE = 60

nTrain = 90
im_width = 400
im_height = 400
test_size = 0.2
num_dim = 3 # RGB images
nValid = 10

root_dir = "../../../project_files/data/"
training_dir = root_dir + "training/"
valid_dir = root_dir + "validation/"

imgs, gt_imgs = load_training(training_dir, nTrain)

imgs_val, gt_imgs_val = load_training(valid_dir, nValid)

def patchesAndOverlap(imgs,patch_size, is_gt = False):
    """ Returns all patches concatenated in one big array and all overlap_images"""
    imgs_patches = []
    overlap_images = []
    for i in range(imgs.shape[0]):
        img_patches,overlap_image = img_patch(imgs[i],patch_size)
        if(is_gt):
            img_patches = np.asarray(img_patches).reshape((img_patches.shape[0],patch_size,patch_size))
        # print(img_patches.shape)
        imgs_patches.append(img_patches)
        overlap_images.append(overlap_image)

    imgs_patches = np.asarray(imgs_patches)
    tmpShape = imgs_patches.shape

    if(not is_gt):
        imgs_patches = imgs_patches.reshape(tmpShape[0]*tmpShape[1],tmpShape[2],tmpShape[3],tmpShape[4])
    if(is_gt):
        imgs_patches = imgs_patches.reshape(tmpShape[0]*tmpShape[1],tmpShape[2],tmpShape[3])

    return imgs_patches,np.asarray(overlap_images)


imgs_patches,overlap_imgs = patchesAndOverlap(imgs,PATCH_SIZE)

gt_imgs = np.expand_dims(gt_imgs,axis=3)
gts_patches,overlap_gts = patchesAndOverlap(gt_imgs,PATCH_SIZE,is_gt = True)

gts_patches_one_hot = convert_to_one_hot(gts_patches)

imgs_patches_val,overlap_imgs_val = patchesAndOverlap(imgs_val,PATCH_SIZE)

gt_imgs_val = np.expand_dims(gt_imgs_val,axis=3)
gts_patches_val,overlap_gts_val = patchesAndOverlap(gt_imgs_val,PATCH_SIZE,is_gt = True)

gts_patches_one_hot_val = convert_to_one_hot(gts_patches_val)


# We can now get all the datas for training and validation

epochs = 50
batch_size = 200 # important parameter
# num_classes = 2 It's always 2 a priori, add default to get_unet

input_img = Input((PATCH_SIZE, PATCH_SIZE, num_dim), name='img')

model = get_unet_2_60(input_img, n_filters=16, dropout=0.2, batchnorm=True)

model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=[f1_score])
model.summary()

checkpointName = 'UNET_patch_2_60'
# model.load_weights(checkpointName+'_chkpt.h5')
callbacks = [EarlyStopping(patience=10, verbose=1),
            ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
            ModelCheckpoint(checkpointName+'_chkpt.h5', verbose=1, save_best_only=False, save_weights_only=True)]

model_train = model.fit(imgs_patches, gts_patches_one_hot, batch_size=batch_size,epochs=epochs,callbacks=callbacks,verbose=1,validation_data=(imgs_patches_val, gts_patches_one_hot_val))

copyfile(checkpointName+'_chkpt.h5',checkpointName+'_'+epochs+'epochs.h5')
with open(checkpointName+'.history', 'wb') as file_pi:
    pickle.dump(model_train.history, file_pi)
print("Training done!")
