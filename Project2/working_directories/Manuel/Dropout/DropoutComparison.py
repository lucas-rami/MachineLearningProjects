
# local imports
from patch import *
from img_load import load_training
from img_manipulation import binarize_imgs

from modelUNET_120 import get_unet_120,f1_score

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

file = open("UNET_patch_120_0.1_.history",'rb')
hist_01 = pickle.load(file)
file.close()
file = open("UNET_patch_120_0.2_.history",'rb')
hist_02 = pickle.load(file)
file.close()
file = open("UNET_patch_120_0.3_.history",'rb')
hist_03 = pickle.load(file)
file.close()
file = open("UNET_patch_120_0.4_.history",'rb')
hist_04 = pickle.load(file)
file.close()
file = open("UNET_patch_120_0.5_.history",'rb')
hist_05 = pickle.load(file)
file.close()
file = open("UNET_patch_120_0.6_.history",'rb')
hist_06 = pickle.load(file)
file.close()
file = open("UNET_patch_120_0.7_.history",'rb')
hist_07 = pickle.load(file)
file.close()
file = open("UNET_patch_120_0.8_.history",'rb')
hist_08 = pickle.load(file)
file.close()
file = open("UNET_patch_120_0.9_.history",'rb')
hist_09 = pickle.load(file)
file.close()

num_classes = 1
num_dim = 3

input_img = Input((120, 120, num_dim), name='img')

model01 = get_unet_120(input_img,num_classes=num_classes,n_filters=16, dropout=0.1, batchnorm=True)
model02 = get_unet_120(input_img,num_classes=num_classes,n_filters=16, dropout=0.2, batchnorm=True)
model03 = get_unet_120(input_img,num_classes=num_classes,n_filters=16, dropout=0.3, batchnorm=True)
model04 = get_unet_120(input_img,num_classes=num_classes,n_filters=16, dropout=0.4, batchnorm=True)
model05 = get_unet_120(input_img,num_classes=num_classes,n_filters=16, dropout=0.5, batchnorm=True)
model06 = get_unet_120(input_img,num_classes=num_classes,n_filters=16, dropout=0.6, batchnorm=True)
model07 = get_unet_120(input_img,num_classes=num_classes,n_filters=16, dropout=0.7, batchnorm=True)
model08 = get_unet_120(input_img,num_classes=num_classes,n_filters=16, dropout=0.8, batchnorm=True)
model09 = get_unet_120(input_img,num_classes=num_classes,n_filters=16, dropout=0.9, batchnorm=True)

model01.load_weights('UNET_patch_120_0.1__500epochs.h5')
model02.load_weights('UNET_patch_120_0.2__500epochs.h5')
model03.load_weights('UNET_patch_120_0.3__500epochs.h5')
model04.load_weights('UNET_patch_120_0.4__500epochs.h5')
model05.load_weights('UNET_patch_120_0.5__500epochs.h5')
model06.load_weights('UNET_patch_120_0.6__500epochs.h5')
model07.load_weights('UNET_patch_120_0.7__500epochs.h5')
model08.load_weights('UNET_patch_120_0.8__500epochs.h5')
model09.load_weights('UNET_patch_120_0.9__500epochs.h5')

valid_dir = "../../../project_files/data/validation/"
val_imgs, val_gts = load_training(valid_dir, 100)

imgs_patches,overlap_imgs,nImgs = make_patch_and_flatten(val_imgs,120)

gts_patches,overlap_gts,nImgs = make_patch_and_flatten(val_gts,120)
gts_patches = np.expand_dims((gts_patches>0.5).astype(int),axis=3)

preds_01 = model01.predict(imgs_patches)
preds_02 = model02.predict(imgs_patches)
preds_03 = model03.predict(imgs_patches)
preds_04 = model04.predict(imgs_patches)
preds_05 = model05.predict(imgs_patches)
preds_06 = model06.predict(imgs_patches)
preds_07 = model07.predict(imgs_patches)
preds_08 = model08.predict(imgs_patches)
preds_09 = model09.predict(imgs_patches)

threshold = np.arange(0,0.8,0.05)

best_score_01 = 0
best_score_02 = 0
best_score_03 = 0
best_score_04 = 0
best_score_05 = 0
best_score_06 = 0
best_score_07 = 0
best_score_08 = 0
best_score_09 = 0

sess = tf.InteractiveSession()
for thr in threshold:
    preds_01_bin = binarize_imgs(preds_01, thr)
    score = sess.run(f1_score(gts_patches,preds_01_bin))
    if score >= best_score_01:
        best_score_01 = score
        thr01 = thr

print("Best score for 0.1: " + str(best_score_01) + ", with threshold: " + str(thr01))

for thr in threshold:
    preds_02_bin = binarize_imgs(preds_02, thr)
    score = sess.run(f1_score(gts_patches,preds_02_bin))
    if score >= best_score_02:
        best_score_02 = score
        thr02 = thr

print("Best score for 0.2: " + str(best_score_02) + ", with threshold: " + str(thr02))

for thr in threshold:
    preds_03_bin = binarize_imgs(preds_03, thr)
    score = sess.run(f1_score(gts_patches,preds_03_bin))
    if score >= best_score_03:
        best_score_03 = score
        thr03 = thr

print("Best score for 0.3: " + str(best_score_03) + ", with threshold: " + str(thr03))

for thr in threshold:
    preds_04_bin = binarize_imgs(preds_04, thr)
    score = sess.run(f1_score(gts_patches,preds_04_bin))
    if score >= best_score_04:
        best_score_04 = score
        thr04 = thr

print("Best score for 0.4: " + str(best_score_04) + ", with threshold: " + str(thr04))

for thr in threshold:
    preds_05_bin = binarize_imgs(preds_05, thr)
    score = sess.run(f1_score(gts_patches,preds_05_bin))
    if score >= best_score_05:
        best_score_05 = score
        thr05 = thr

print("Best score for 0.5: " + str(best_score_05) + ", with threshold: " + str(thr05))

for thr in threshold:
    preds_06_bin = binarize_imgs(preds_06, thr)
    score = sess.run(f1_score(gts_patches,preds_06_bin))
    if score >= best_score_06:
        best_score_06 = score
        thr06 = thr

print("Best score for 0.6: " + str(best_score_06) + ", with threshold: " + str(thr06))

for thr in threshold:
    preds_07_bin = binarize_imgs(preds_07, thr)
    score = sess.run(f1_score(gts_patches,preds_07_bin))
    if score >= best_score_07:
        best_score_07 = score
        thr07 = thr

print("Best score for 0.7: " + str(best_score_07) + ", with threshold: " + str(thr07))

for thr in threshold:
    preds_08_bin = binarize_imgs(preds_08, thr)
    score = sess.run(f1_score(gts_patches,preds_08_bin))
    if score >= best_score_08:
        best_score_08 = score
        thr08 = thr

print("Best score for 0.8: " + str(best_score_08) + ", with threshold: " + str(thr08))

for thr in threshold:
    preds_09_bin = binarize_imgs(preds_09, thr)
    score = sess.run(f1_score(gts_patches,preds_09_bin))
    if score >= best_score_09:
        best_score_09 = score
        thr09 = thr

print("Best score for 0.9: " + str(best_score_09) + ", with threshold: " + str(thr09))

sess.close()

preds_01_bin = binarize_imgs(preds_01, thr01)
preds_02_bin = binarize_imgs(preds_02, thr02)
preds_03_bin = binarize_imgs(preds_03, thr03)
preds_04_bin = binarize_imgs(preds_04, thr04)
preds_05_bin = binarize_imgs(preds_05, thr05)
preds_06_bin = binarize_imgs(preds_06, thr06)
preds_07_bin = binarize_imgs(preds_07, thr07)
preds_08_bin = binarize_imgs(preds_08, thr08)
preds_09_bin = binarize_imgs(preds_09, thr09)

plt.imshow(imgs_patches[65])
plt.imshow(preds_05_bin[65,:,:,0])

fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
model_01, = ax.plot(hist_01['val_f1_score'], label='0.1')
model_02, = ax.plot(hist_02['val_f1_score'], label='0.2')
model_03, = ax.plot(hist_03['val_f1_score'], label='0.3')
model_04, = ax.plot(hist_04['val_f1_score'], label='0.4')
model_05, = ax.plot(hist_05['val_f1_score'], label='0.5')
model_06, = ax.plot(hist_06['val_f1_score'], label='0.6')
model_07, = ax.plot(hist_07['val_f1_score'], label='0.7')
model_08, = ax.plot(hist_08['val_f1_score'], label='0.8')
model_09, = ax.plot(hist_09['val_f1_score'], label='0.9')
plt.ylabel('F1 score')
plt.xlabel('Epochs')
plt.legend(handles=[model_01,model_02,model_03,model_04,model_05,model_06,model_07,model_08,model_09])
plt.show()
fig.savefig('dropout_f1.pdf', bbox_inches='tight')   # save the figure to file
plt.close(fig)

fig2, ax2 = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
model_01, = ax2.plot(hist_01['val_acc'], label='0.1')
model_02, = ax2.plot(hist_02['val_acc'], label='0.2')
model_03, = ax2.plot(hist_03['val_acc'], label='0.3')
model_04, = ax2.plot(hist_04['val_acc'], label='0.4')
model_05, = ax2.plot(hist_05['val_acc'], label='0.5')
model_06, = ax2.plot(hist_06['val_acc'], label='0.6')
model_07, = ax2.plot(hist_07['val_acc'], label='0.7')
model_08, = ax2.plot(hist_08['val_acc'], label='0.8')
model_09, = ax2.plot(hist_09['val_acc'], label='0.9')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(handles=[model_01,model_02,model_03,model_04,model_05,model_06,model_07,model_08,model_09])
plt.show()
fig2.savefig('dropout_acc.pdf', bbox_inches='tight')   # save the figure to file
plt.close(fig2)

fig3, ax3 = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
model_01, = ax3.plot(hist_01['val_loss'], label='0.1')
model_02, = ax3.plot(hist_02['val_loss'], label='0.2')
model_03, = ax3.plot(hist_03['val_loss'], label='0.3')
model_04, = ax3.plot(hist_04['val_loss'], label='0.4')
model_05, = ax3.plot(hist_05['val_loss'], label='0.5')
model_06, = ax3.plot(hist_06['val_loss'], label='0.6')
model_07, = ax3.plot(hist_07['val_loss'], label='0.7')
model_08, = ax3.plot(hist_08['val_loss'], label='0.8')
model_09, = ax3.plot(hist_09['val_loss'], label='0.9')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(handles=[model_01,model_02,model_03,model_04,model_05,model_06,model_07,model_08,model_09])
plt.show()
fig3.savefig('dropout_loss.pdf', bbox_inches='tight')   # save the figure to file
plt.close(fig3)
