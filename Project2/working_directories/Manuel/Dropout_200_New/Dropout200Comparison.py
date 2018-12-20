
# local imports
import patch
import load
import transformation as tr

from modelCNN_200 import get_unet,f1_score

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

def f1_custom(labels, preds):
    true_positives = np.sum(labels*preds)
    try:
        precision = true_positives / np.sum(preds)
        recall = true_positives / np.sum(labels)
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        return 0.0
    return f1

file = open("CNN_200_rot_dropout0.2.history",'rb')
hist_02 = pickle.load(file)
file.close()
file = open("CNN_200_rot_dropout0.5.history",'rb')
hist_05 = pickle.load(file)
file.close()
file = open("CNN_200_rot_dropout0.6.history",'rb')
hist_06 = pickle.load(file)
file.close()
file = open("CNN_200_rot_dropout0.9.history",'rb')
hist_09 = pickle.load(file)
file.close()

num_classes = 1
num_dim = 3

input_img = Input((200, 200, num_dim), name='img')

model01 = get_unet(input_img,num_classes=num_classes,n_filters=16, dropout=0.1, batchnorm=True)
model02 = get_unet(input_img,num_classes=num_classes,n_filters=16, dropout=0.2, batchnorm=True)
model03 = get_unet(input_img,num_classes=num_classes,n_filters=16, dropout=0.3, batchnorm=True)
model04 = get_unet(input_img,num_classes=num_classes,n_filters=16, dropout=0.4, batchnorm=True)
model05 = get_unet(input_img,num_classes=num_classes,n_filters=16, dropout=0.5, batchnorm=True)
model06 = get_unet(input_img,num_classes=num_classes,n_filters=16, dropout=0.6, batchnorm=True)
model07 = get_unet(input_img,num_classes=num_classes,n_filters=16, dropout=0.7, batchnorm=True)
model08 = get_unet(input_img,num_classes=num_classes,n_filters=16, dropout=0.8, batchnorm=True)
model09 = get_unet(input_img,num_classes=num_classes,n_filters=16, dropout=0.9, batchnorm=True)

model01.load_weights('test_CNN_200_rot_dropout0.1.h5')
model02.load_weights('test_CNN_200_rot_dropout0.2.h5')
model03.load_weights('test_CNN_200_rot_dropout0.3.h5')
model04.load_weights('test_CNN_200_rot_dropout0.4.h5')
model05.load_weights('test_CNN_200_rot_dropout0.5.h5')
model06.load_weights('test_CNN_200_rot_dropout0.6.h5')
model07.load_weights('test_CNN_200_rot_dropout0.7.h5')
model08.load_weights('test_CNN_200_rot_dropout0.8.h5')
model09.load_weights('test_CNN_200_rot_dropout0.9.h5')

valid_dir = "../../../project_files/data/validation/"
val_imgs, val_gts = load.load_training_data(valid_dir)
val_gts = np.expand_dims(val_gts, axis=3)

resize_threshold = 0.25
height = 200
width = 200

resized_val_imgs = tr.resize_imgs(val_imgs, height, width)
resized_val_gts = tr.resize_binary_imgs(val_gts, height, width, resize_threshold)

preds_01 = model01.predict(resized_val_imgs)
preds_02 = model02.predict(resized_val_imgs)
preds_03 = model03.predict(resized_val_imgs)
preds_04 = model04.predict(resized_val_imgs)
preds_05 = model05.predict(resized_val_imgs)
preds_06 = model06.predict(resized_val_imgs)
preds_07 = model07.predict(resized_val_imgs)
preds_08 = model08.predict(resized_val_imgs)
preds_09 = model09.predict(resized_val_imgs)

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

for thr in threshold:
    preds_01_bin = (preds_01 > thr).astype(int)
    score = f1_custom(resized_val_gts,preds_01_bin)
    if score >= best_score_01:
        best_score_01 = score
        thr01 = thr

print("Best score for 0.1: " + str(best_score_01) + ", with threshold: " + str(thr01))

for thr in threshold:
    preds_02_bin = (preds_02 > thr).astype(int)
    score = f1_custom(resized_val_gts,preds_02_bin)
    if score >= best_score_02:
        best_score_02 = score
        thr02 = thr

print("Best score for 0.2: " + str(best_score_02) + ", with threshold: " + str(thr02))

for thr in threshold:
    preds_03_bin = (preds_03 > thr).astype(int)
    score = f1_custom(resized_val_gts,preds_03_bin)
    if score >= best_score_03:
        best_score_03 = score
        thr03 = thr

print("Best score for 0.3: " + str(best_score_03) + ", with threshold: " + str(thr03))

for thr in threshold:
    preds_04_bin = (preds_04 > thr).astype(int)
    score = f1_custom(resized_val_gts,preds_04_bin)
    if score >= best_score_04:
        best_score_04 = score
        thr04 = thr

print("Best score for 0.4: " + str(best_score_04) + ", with threshold: " + str(thr04))

for thr in threshold:
    preds_05_bin = (preds_05 > thr).astype(int)
    score = f1_custom(resized_val_gts,preds_05_bin)
    if score >= best_score_05:
        best_score_05 = score
        thr05 = thr

print("Best score for 0.5: " + str(best_score_05) + ", with threshold: " + str(thr05))

for thr in threshold:
    preds_06_bin = (preds_06 > thr).astype(int)
    score = f1_custom(resized_val_gts,preds_06_bin)
    if score >= best_score_06:
        best_score_06 = score
        thr06 = thr

print("Best score for 0.6: " + str(best_score_06) + ", with threshold: " + str(thr06))

for thr in threshold:
    preds_07_bin = (preds_07 > thr).astype(int)
    score = f1_custom(resized_val_gts,preds_07_bin)
    if score >= best_score_07:
        best_score_07 = score
        thr07 = thr

print("Best score for 0.7: " + str(best_score_07) + ", with threshold: " + str(thr07))

for thr in threshold:
    preds_08_bin = (preds_08 > thr).astype(int)
    score = f1_custom(resized_val_gts,preds_08_bin)
    if score >= best_score_08:
        best_score_08 = score
        thr08 = thr

print("Best score for 0.8: " + str(best_score_08) + ", with threshold: " + str(thr08))

for thr in threshold:
    preds_09_bin = (preds_09 > thr).astype(int)
    score = f1_custom(resized_val_gts,preds_09_bin)
    if score >= best_score_09:
        best_score_09 = score
        thr09 = thr

print("Best score for 0.9: " + str(best_score_09) + ", with threshold: " + str(thr09))

Best score for 0.1: 0.894930159270931, with threshold: 0.45
Best score for 0.2: 0.6232529412350698, with threshold: 0.5
Best score for 0.3: 0.8375024805991039, with threshold: 0.45
Best score for 0.4: 0.8376818866031108, with threshold: 0.4
Best score for 0.5: 0.7767882213945506, with threshold: 0.55
Best score for 0.6: 0.8910252065792561, with threshold: 0.5
Best score for 0.7: 0.8405458912968878, with threshold: 0.6000000000000001
Best score for 0.8: 0.7781450613240347, with threshold: 0.55
Best score for 0.9: 0.6453467964899627, with threshold: 0.45

preds_01_bin = (preds_01 > thr01).astype(int)
preds_02_bin = (preds_02 > thr02).astype(int)
preds_03_bin = (preds_03 > thr03).astype(int)
preds_04_bin = (preds_04 > thr04).astype(int)
preds_05_bin = (preds_05 > thr05).astype(int)
preds_06_bin = (preds_06 > thr06).astype(int)
preds_07_bin = (preds_07 > thr07).astype(int)
preds_08_bin = (preds_08 > thr08).astype(int)
preds_09_bin = (preds_09 > thr09).astype(int)

plt.imshow(val_imgs[8])
plt.imshow(preds_05_bin[8,:,:,0])



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
