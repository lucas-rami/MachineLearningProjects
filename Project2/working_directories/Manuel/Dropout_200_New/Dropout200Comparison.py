
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

model01.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=[f1_score, 'accuracy'])
model02.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=[f1_score, 'accuracy'])
model03.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=[f1_score, 'accuracy'])
model04.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=[f1_score, 'accuracy'])
model05.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=[f1_score, 'accuracy'])
model06.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=[f1_score, 'accuracy'])
model07.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=[f1_score, 'accuracy'])
model08.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=[f1_score, 'accuracy'])
model09.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=[f1_score, 'accuracy'])

resize_threshold = 0.25
height = 200
width = 200

resized_val_imgs = tr.resize_imgs(val_imgs, height, width)
resized_val_gts = tr.resize_binary_imgs(val_gts, height, width, resize_threshold)

perf01 = model01.evaluate(resized_val_imgs,resized_val_gts)
perf02 = model02.evaluate(resized_val_imgs,resized_val_gts)
perf03 = model03.evaluate(resized_val_imgs,resized_val_gts)
perf04 = model04.evaluate(resized_val_imgs,resized_val_gts)
perf05 = model05.evaluate(resized_val_imgs,resized_val_gts)
perf06 = model06.evaluate(resized_val_imgs,resized_val_gts)
perf07 = model07.evaluate(resized_val_imgs,resized_val_gts)
perf08 = model08.evaluate(resized_val_imgs,resized_val_gts)
perf09 = model09.evaluate(resized_val_imgs,resized_val_gts)

dropouts = np.arange(0.1,1,0.1)

losses = []
losses.append(perf01[0])
losses.append(perf02[0])
losses.append(perf03[0])
losses.append(perf04[0])
losses.append(perf05[0])
losses.append(perf06[0])
losses.append(perf07[0])
losses.append(perf08[0])
losses.append(perf09[0])

fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
model_01, = ax.plot(dropouts,losses)
plt.ylabel('Loss')
plt.xlabel('Dropout')
plt.show()
fig.savefig('dropout_200_loss.pdf', bbox_inches='tight')   # save the figure to file
plt.close(fig)
