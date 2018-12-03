# %%
from keras.layers import Dense, Conv2D, Flatten
from keras.models import Sequential
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.datasets import mnist
# download mnist data and split into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# %%
# plot the first image in the dataset
plt.imshow(X_train[0])


# %%
# check image shape
X_train.shape

# %%
# reshape data to fit model
X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)

# %%

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# %%
# create model
model = Sequential()

# add model layers
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

#compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#%%
#train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)

#%%
