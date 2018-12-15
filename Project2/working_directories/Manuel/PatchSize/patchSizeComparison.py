
# local imports
from patch import img_patch
from img_load import load_training
from img_manipulation import binarize_imgs

from modelUNET_120 import get_unet_120,f1_score
from modelUNET_100 import get_unet_100
from modelUNET_80 import get_unet_80
from modelUNET_60 import get_unet_60
from modelUNET_40 import get_unet_40
from modelUNET import get_unet_80_bl

# librairies imports
import numpy as np
import matplotlib.pyplot as plt
import pickle
from keras.models import *
from keras.layers import *
from keras.regularizers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam

file = open("UNET_patch_120.history",'rb')
hist_120 = pickle.load(file)
file.close()
file = open("UNET_patch_100.history",'rb')
hist_100 = pickle.load(file)
file.close()
file = open("UNET_patch_80.history",'rb')
hist_80 = pickle.load(file)
file.close()
file = open("UNET_patch_60.history",'rb')
hist_60 = pickle.load(file)
file.close()
file = open("UNET_patch_40.history",'rb')
hist_40 = pickle.load(file)
file.close()
file = open("UNET_patch_80_bl.history",'rb')
hist_80_baseline = pickle.load(file)
file.close()

num_classes = 1
num_dim = 3

input_img_40 = Input((40, 40, num_dim), name='img')
input_img_60 = Input((60, 60, num_dim), name='img')
input_img_80 = Input((80, 80, num_dim), name='img')
input_img_100 = Input((100, 100, num_dim), name='img')
input_img_120 = Input((120, 120, num_dim), name='img')

model40 = get_unet_40(input_img_40,num_classes=num_classes,n_filters=16, dropout=0.5, batchnorm=True)
model60 = get_unet_60(input_img_60,num_classes=num_classes,n_filters=16, dropout=0.5, batchnorm=True)
model80 = get_unet_80(input_img_80,num_classes=num_classes,n_filters=16, dropout=0.5, batchnorm=True)
model80bl = get_unet_80_bl(input_img_80,num_classes=num_classes,n_filters=16, dropout=0.5, batchnorm=True)
model100 = get_unet_100(input_img_100,num_classes=num_classes,n_filters=16, dropout=0.5, batchnorm=True)
model120 = get_unet_120(input_img_120,num_classes=num_classes,n_filters=16, dropout=0.5, batchnorm=True)

model40.load_weights('UNET_patch_40_50epochs.h5')
model60.load_weights('UNET_patch_60_50epochs.h5')
model80.load_weights('UNET_patch_80_50epochs.h5')
model80bl.load_weights('UNET_patch_80_bl_50epochs.h5')
model100.load_weights('UNET_patch_100_50epochs.h5')
model120.load_weights('UNET_patch_120_50epochs.h5')

valid_dir = "../../../project_files/data/validation/"
val_imgs, val_gts = load_training(valid_dir, 100)

def patchesAndOverlap(imgs,patch_size,overlap=0,is_gt = False):
    """ Returns all patches concatenated in one big array and all overlap_images"""
    imgs_patches = []
    overlap_images = []
    for i in range(imgs.shape[0]):
        img_patches,overlap_image = img_patch(imgs[i],patch_size,overlap,overlap)
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

imgs_patches_40,overlap_imgs_40 = patchesAndOverlap(val_imgs,40)
imgs_patches_60,overlap_imgs_60 = patchesAndOverlap(val_imgs,60)
imgs_patches_80,overlap_imgs_80 = patchesAndOverlap(val_imgs,80)
imgs_patches_100,overlap_imgs_100 = patchesAndOverlap(val_imgs,100)
imgs_patches_120,overlap_imgs_120 = patchesAndOverlap(val_imgs,120)

overlap = 0

gts_patches_40,overlap_gts_40 = patchesAndOverlap(val_gts,40,overlap,is_gt = True)
gts_patches_60,overlap_gts_60 = patchesAndOverlap(val_gts,60,overlap,is_gt = True)
gts_patches_80,overlap_gts_80 = patchesAndOverlap(val_gts,80,overlap,is_gt = True)
gts_patches_100,overlap_gts_100 = patchesAndOverlap(val_gts,100,overlap,is_gt = True)
gts_patches_120,overlap_gts_120 = patchesAndOverlap(val_gts,120,overlap,is_gt = True)
gts_patches_40 = np.expand_dims((gts_patches_40>0.5).astype(int),axis=3)
gts_patches_60 = np.expand_dims((gts_patches_60>0.5).astype(int),axis=3)
gts_patches_80 = np.expand_dims((gts_patches_80>0.5).astype(int),axis=3)
gts_patches_100 = np.expand_dims((gts_patches_100>0.5).astype(int),axis=3)
gts_patches_120 = np.expand_dims((gts_patches_120>0.5).astype(int),axis=3)

preds_40 = model40.predict(imgs_patches_40)
preds_60 = model60.predict(imgs_patches_60)
preds_80 = model80.predict(imgs_patches_80)
preds_80_bl = model80bl.predict(imgs_patches_80)
preds_100 = model100.predict(imgs_patches_100)
preds_120 = model120.predict(imgs_patches_120)

threshold = np.arange(0.01,0.51,0.01)

plt.imshow(preds_40[1,:,:,0])
plt.imshow(preds_40_bin[1,:,:,0])

fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
model_120, = ax.plot(hist_120['val_f1_score'], label='120')
model_100, = ax.plot(hist_100['val_f1_score'], label='100')
model_80, = ax.plot(hist_80['val_f1_score'], label='80')
model_60, = ax.plot(hist_60['val_f1_score'], label='60')
model_40, = ax.plot(hist_40['val_f1_score'], label='40')
model_80_bl, = ax.plot(hist_80_baseline['val_f1_score'], label='80, Phil')
plt.ylabel('F1 score')
plt.xlabel('Epochs')
plt.legend(handles=[model_120,model_100,model_80,model_60,model_40,model_80_bl])
plt.show()
fig.savefig('patchSize_f1.pdf', bbox_inches='tight')   # save the figure to file
plt.close(fig)

fig2, ax2 = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
model_120, = ax2.plot(hist_120['val_acc'], label='120')
model_100, = ax2.plot(hist_100['val_acc'], label='100')
model_80, = ax2.plot(hist_80['val_acc'], label='80')
model_60, = ax2.plot(hist_60['val_acc'], label='60')
model_40, = ax2.plot(hist_40['val_acc'], label='40')
model_80_bl, = ax2.plot(hist_80_baseline['val_acc'], label='80, Phil')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(handles=[model_120,model_100,model_80,model_60,model_40,model_80_bl])
plt.show()
fig2.savefig('patchSize_acc.pdf', bbox_inches='tight')   # save the figure to file
plt.close(fig2)

fig3, ax3 = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
model_120, = ax3.plot(hist_120['val_loss'], label='120')
model_100, = ax3.plot(hist_100['val_loss'], label='100')
model_80, = ax3.plot(hist_80['val_loss'], label='80')
model_60, = ax3.plot(hist_60['val_loss'], label='60')
model_40, = ax3.plot(hist_40['val_loss'], label='40')
model_80_bl, = ax3.plot(hist_80_baseline['val_loss'], label='80, Phil')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(handles=[model_120,model_100,model_80,model_60,model_40,model_80_bl])
plt.show()
fig3.savefig('patchSize_loss.pdf', bbox_inches='tight')   # save the figure to file
plt.close(fig3)
