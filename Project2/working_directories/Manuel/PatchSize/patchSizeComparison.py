
# local imports
from patch import img_patch
from img_load import load_training
from img_manipulation import binarize_imgs

from modelUNET_200 import get_unet_200
from modelUNET_180 import get_unet_180
from modelUNET_160 import get_unet_160
from modelUNET_140 import get_unet_140
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
import tensorflow as tf
from keras.models import *
from keras.layers import *
from keras.regularizers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam

file = open("UNET_patch_200.history",'rb')
hist_200 = pickle.load(file)
file.close()
file = open("UNET_patch_180.history",'rb')
hist_180 = pickle.load(file)
file.close()
file = open("UNET_patch_160.history",'rb')
hist_160 = pickle.load(file)
file.close()
file = open("UNET_patch_140.history",'rb')
hist_140 = pickle.load(file)
file.close()
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
input_img_140 = Input((140, 140, num_dim), name='img')
input_img_160 = Input((160, 160, num_dim), name='img')
input_img_180 = Input((180, 180, num_dim), name='img')
input_img_200 = Input((200, 200, num_dim), name='img')

model40 = get_unet_40(input_img_40,num_classes=num_classes,n_filters=16, dropout=0.5, batchnorm=True)
model60 = get_unet_60(input_img_60,num_classes=num_classes,n_filters=16, dropout=0.5, batchnorm=True)
model80 = get_unet_80(input_img_80,num_classes=num_classes,n_filters=16, dropout=0.5, batchnorm=True)
model80bl = get_unet_80_bl(input_img_80,num_classes=num_classes,n_filters=16, dropout=0.5, batchnorm=True)
model100 = get_unet_100(input_img_100,num_classes=num_classes,n_filters=16, dropout=0.5, batchnorm=True)
model120 = get_unet_120(input_img_120,num_classes=num_classes,n_filters=16, dropout=0.5, batchnorm=True)
model140 = get_unet_140(input_img_140,num_classes=num_classes,n_filters=16, dropout=0.5, batchnorm=True)
model160 = get_unet_160(input_img_160,num_classes=num_classes,n_filters=16, dropout=0.5, batchnorm=True)
model180 = get_unet_180(input_img_180,num_classes=num_classes,n_filters=16, dropout=0.5, batchnorm=True)
model200 = get_unet_200(input_img_200,num_classes=num_classes,n_filters=16, dropout=0.5, batchnorm=True)

model40.load_weights('UNET_patch_40_50epochs.h5')
model60.load_weights('UNET_patch_60_50epochs.h5')
model80.load_weights('UNET_patch_80_50epochs.h5')
model80bl.load_weights('UNET_patch_80_bl_50epochs.h5')
model100.load_weights('UNET_patch_100_50epochs.h5')
model120.load_weights('UNET_patch_120_50epochs.h5')
model140.load_weights('UNET_patch_140_500epochs.h5')
model160.load_weights('UNET_patch_160_500epochs.h5')
model180.load_weights('UNET_patch_180_500epochs.h5')
model200.load_weights('UNET_patch_200_500epochs.h5')

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
imgs_patches_140,overlap_imgs_140 = patchesAndOverlap(val_imgs,140)
imgs_patches_160,overlap_imgs_160 = patchesAndOverlap(val_imgs,160)
imgs_patches_180,overlap_imgs_180 = patchesAndOverlap(val_imgs,180)
imgs_patches_200,overlap_imgs_200 = patchesAndOverlap(val_imgs,200)

overlap = 0

gts_patches_40,overlap_gts_40 = patchesAndOverlap(val_gts,40,overlap,is_gt = True)
gts_patches_60,overlap_gts_60 = patchesAndOverlap(val_gts,60,overlap,is_gt = True)
gts_patches_80,overlap_gts_80 = patchesAndOverlap(val_gts,80,overlap,is_gt = True)
gts_patches_100,overlap_gts_100 = patchesAndOverlap(val_gts,100,overlap,is_gt = True)
gts_patches_120,overlap_gts_120 = patchesAndOverlap(val_gts,120,overlap,is_gt = True)
gts_patches_140,overlap_gts_140 = patchesAndOverlap(val_gts,140,overlap,is_gt = True)
gts_patches_160,overlap_gts_160 = patchesAndOverlap(val_gts,160,overlap,is_gt = True)
gts_patches_180,overlap_gts_180 = patchesAndOverlap(val_gts,180,overlap,is_gt = True)
gts_patches_200,overlap_gts_200 = patchesAndOverlap(val_gts,200,overlap,is_gt = True)
gts_patches_40 = np.expand_dims((gts_patches_40>0.5).astype(int),axis=3)
gts_patches_60 = np.expand_dims((gts_patches_60>0.5).astype(int),axis=3)
gts_patches_80 = np.expand_dims((gts_patches_80>0.5).astype(int),axis=3)
gts_patches_100 = np.expand_dims((gts_patches_100>0.5).astype(int),axis=3)
gts_patches_120 = np.expand_dims((gts_patches_120>0.5).astype(int),axis=3)
gts_patches_140 = np.expand_dims((gts_patches_140>0.5).astype(int),axis=3)
gts_patches_160 = np.expand_dims((gts_patches_160>0.5).astype(int),axis=3)
gts_patches_180 = np.expand_dims((gts_patches_180>0.5).astype(int),axis=3)
gts_patches_200 = np.expand_dims((gts_patches_200>0.5).astype(int),axis=3)

preds_40 = model40.predict(imgs_patches_40)
preds_60 = model60.predict(imgs_patches_60)
preds_80 = model80.predict(imgs_patches_80)
preds_80_bl = model80bl.predict(imgs_patches_80)
preds_100 = model100.predict(imgs_patches_100)
preds_120 = model120.predict(imgs_patches_120)
preds_140 = model140.predict(imgs_patches_140)
preds_160 = model160.predict(imgs_patches_160)
preds_180 = model180.predict(imgs_patches_180)
preds_200 = model200.predict(imgs_patches_200)

threshold = np.arange(0,0.8,0.05)

best_score_40 = 0
best_score_60 = 0
best_score_80 = 0
best_score_80_bl = 0
best_score_100 = 0
best_score_120 = 0
best_score_140 = 0
best_score_160 = 0
best_score_180 = 0
best_score_200 = 0

sess = tf.InteractiveSession()
for thr in threshold:
    preds_40_bin = binarize_imgs(preds_40, thr)
    score = sess.run(f1_score(gts_patches_40,preds_40_bin))
    if score >= best_score_40:
        best_score_40 = score
        thr40 = thr

print("Best score for 40: " + str(best_score_40) + ", with threshold: " + str(thr40))

for thr in threshold:
    preds_60_bin = binarize_imgs(preds_60, thr)
    score = sess.run(f1_score(gts_patches_60,preds_60_bin))
    if score >= best_score_60:
        best_score_60 = score
        thr60 = thr

print("Best score for 60: " + str(best_score_60) + ", with threshold: " + str(thr60))

for thr in threshold:
    preds_80_bin = binarize_imgs(preds_80, thr)
    score = sess.run(f1_score(gts_patches_80,preds_80_bin))
    if score >= best_score_80:
        best_score_80 = score
        thr80 = thr

print("Best score for 80: " + str(best_score_80) + ", with threshold: " + str(thr80))

for thr in threshold:
    preds_80_bl_bin = binarize_imgs(preds_80_bl, thr)
    score = sess.run(f1_score(gts_patches_80,preds_80_bl_bin))
    if score >= best_score_80_bl:
        best_score_80_bl = score
        thr80_bl = thr

print("Best score for 80, baseline: " + str(best_score_80_bl) + ", with threshold: " + str(thr80_bl))

for thr in threshold:
    preds_100_bin = binarize_imgs(preds_100, thr)
    score = sess.run(f1_score(gts_patches_100,preds_100_bin))
    if score >= best_score_100:
        best_score_100 = score
        thr100 = thr

print("Best score for 100: " + str(best_score_100) + ", with threshold: " + str(thr100))

for thr in threshold:
    preds_120_bin = binarize_imgs(preds_120, thr)
    score = sess.run(f1_score(gts_patches_120,preds_120_bin))
    if score >= best_score_120:
        best_score_120 = score
        thr120 = thr

print("Best score for 120: " + str(best_score_120) + ", with threshold: " + str(thr120))

for thr in threshold:
    preds_140_bin = binarize_imgs(preds_140, thr)
    score = sess.run(f1_score(gts_patches_140,preds_140_bin))
    if score >= best_score_140:
        best_score_140 = score
        thr140 = thr

print("Best score for 140: " + str(best_score_140) + ", with threshold: " + str(thr140))

for thr in threshold:
    preds_160_bin = binarize_imgs(preds_160, thr)
    score = sess.run(f1_score(gts_patches_160,preds_160_bin))
    if score >= best_score_160:
        best_score_160 = score
        thr160 = thr

print("Best score for 160: " + str(best_score_160) + ", with threshold: " + str(thr160))

for thr in threshold:
    preds_180_bin = binarize_imgs(preds_180, thr)
    score = sess.run(f1_score(gts_patches_180,preds_180_bin))
    if score >= best_score_180:
        best_score_180 = score
        thr180 = thr

print("Best score for 180: " + str(best_score_180) + ", with threshold: " + str(thr180))

for thr in threshold:
    preds_200_bin = binarize_imgs(preds_200, thr)
    score = sess.run(f1_score(gts_patches_200,preds_200_bin))
    if score >= best_score_200:
        best_score_200 = score
        thr200 = thr

print("Best score for 200: " + str(best_score_200) + ", with threshold: " + str(thr200))

sess.close()

preds_40_bin = binarize_imgs(preds_40, thr40)
preds_60_bin = binarize_imgs(preds_60, thr60)
preds_80_bin = binarize_imgs(preds_80, thr80)
preds_80_bl_bin = binarize_imgs(preds_80_bl, thr80_bl)
preds_100_bin = binarize_imgs(preds_100, thr100)
preds_120_bin = binarize_imgs(preds_120, thr120)
preds_140_bin = binarize_imgs(preds_140, thr140)
preds_160_bin = binarize_imgs(preds_160, thr160)
preds_180_bin = binarize_imgs(preds_180, thr180)
preds_200_bin = binarize_imgs(preds_200, thr200)

plt.imshow(imgs_patches_200[5])
plt.imshow(preds_200_bin[5,:,:,0])

fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
model_200, = ax.plot(hist_200['val_f1_score'], label='200')
model_180, = ax.plot(hist_180['val_f1_score'], label='180')
model_160, = ax.plot(hist_160['val_f1_score'], label='160')
model_140, = ax.plot(hist_140['val_f1_score'], label='140')
model_120, = ax.plot(hist_120['val_f1_score'], label='120')
model_100, = ax.plot(hist_100['val_f1_score'], label='100')
model_80, = ax.plot(hist_80['val_f1_score'], label='80')
model_60, = ax.plot(hist_60['val_f1_score'], label='60')
model_40, = ax.plot(hist_40['val_f1_score'], label='40')
model_80_bl, = ax.plot(hist_80_baseline['val_f1_score'], label='80, Phil')
plt.ylabel('F1 score')
plt.xlabel('Epochs')
plt.legend(handles=[model_200,model_180,model_160,model_140,model_120,model_100,model_80,model_60,model_40,model_80_bl])
plt.show()
fig.savefig('patchSize_f1.pdf', bbox_inches='tight')   # save the figure to file
plt.close(fig)

fig2, ax2 = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
model_200, = ax2.plot(hist_200['val_acc'], label='200')
model_180, = ax2.plot(hist_180['val_acc'], label='180')
model_160, = ax2.plot(hist_160['val_acc'], label='160')
model_140, = ax2.plot(hist_140['val_acc'], label='140')
model_120, = ax2.plot(hist_120['val_acc'], label='120')
model_100, = ax2.plot(hist_100['val_acc'], label='100')
model_80, = ax2.plot(hist_80['val_acc'], label='80')
model_60, = ax2.plot(hist_60['val_acc'], label='60')
model_40, = ax2.plot(hist_40['val_acc'], label='40')
model_80_bl, = ax2.plot(hist_80_baseline['val_acc'], label='80, Phil')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(handles=[model_200,model_180,model_160,model_140,model_120,model_100,model_80,model_60,model_40,model_80_bl])
plt.show()
fig2.savefig('patchSize_acc.pdf', bbox_inches='tight')   # save the figure to file
plt.close(fig2)

fig3, ax3 = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
model_200, = ax3.plot(hist_200['val_loss'], label='200')
model_180, = ax3.plot(hist_180['val_loss'], label='180')
model_160, = ax3.plot(hist_160['val_loss'], label='160')
model_140, = ax3.plot(hist_140['val_loss'], label='140')
model_120, = ax3.plot(hist_120['val_loss'], label='120')
model_100, = ax3.plot(hist_100['val_loss'], label='100')
model_80, = ax3.plot(hist_80['val_loss'], label='80')
model_60, = ax3.plot(hist_60['val_loss'], label='60')
model_40, = ax3.plot(hist_40['val_loss'], label='40')
model_80_bl, = ax3.plot(hist_80_baseline['val_loss'], label='80, Phil')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(handles=[model_200,model_180,model_160,model_140,model_120,model_100,model_80,model_60,model_40,model_80_bl])
plt.show()
fig3.savefig('patchSize_loss.pdf', bbox_inches='tight')   # save the figure to file
plt.close(fig3)
