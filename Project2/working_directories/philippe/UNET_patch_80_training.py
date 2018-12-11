
# local imports
from patch import img_patch
from img_load import load_training
from img_manipulation import convert_to_one_hot

from modelUNET import get_unet,f1_score

# librairies imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import *
from keras.layers import *
from keras.regularizers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam

PATCH_SIZE = 80

nTrain = 90
im_width = 400
im_height = 400
test_size = 0.2
num_dim = 3 # RGB images

root_dir = "../../project_files/data/"
training_dir = root_dir + "training/"

imgs, gt_imgs = load_training(training_dir, nTrain)


def patchesAndOverlap(imgs,patch_size, is_gt = False):
    """ Returns all patches concatenated in one big array and all overlap_images"""
    imgs_patches = []
    overlap_images = []
    for i in range(0,nTrain):
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

gt_imgs = gt_imgs.reshape((nTrain,im_height,im_width,1))
gts_patches,overlap_gts = patchesAndOverlap(gt_imgs,PATCH_SIZE,is_gt = True)

gts_patches_one_hot = convert_to_one_hot(gts_patches)


# We can now get all the datas for training and validation
X_train, X_valid, y_train, y_valid = train_test_split(imgs_patches, gts_patches_one_hot, test_size=test_size)

epochs = 1
batch_size = 200 # important parameter
# num_classes = 2 It's always 2 a priori, add default to get_unet

input_img = Input((PATCH_SIZE, PATCH_SIZE, num_dim), name='img')

model = get_unet(input_img, n_filters=64, dropout=0.2, batchnorm=False)

model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=[f1_score])
model.summary()

checkpointName = 'UNET_patch_80'
# model.load_weights(checkpointName+'_chkpt.h5')
callbacks = [EarlyStopping(patience=10, verbose=1),
            ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
            ModelCheckpoint(checkpointName+'_chkpt.h5', verbose=1, save_best_only=False, save_weights_only=True)]

model_train = model.fit(X_train, y_train, batch_size=batch_size,epochs=epochs,callbacks=callbacks,verbose=1,validation_data=(X_valid, y_valid))

copyfile(checkpointName+'_chkpt.h5',checkpointName+'_'+epochs+'epochs'.h5')
