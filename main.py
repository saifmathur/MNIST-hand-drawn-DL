# -*- coding: utf-8 -*-
"""
Created on Mon May 11 16:58:37 2020

@author: Saif Mathur
"""

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

(xtrain,ytrain),(xtest,ytest) = mnist.load_data()


xtrain = xtrain.reshape(xtrain.shape[0],28,28,1)
xtest = xtest.reshape(xtest.shape[0],28,28,1)

input_shape = (28,28,1)

ytrain = keras.utils.to_categorical(ytrain, 10) 
ytest = keras.utils.to_categorical(ytest,10)

xtrain = xtrain.astype('float32')
xtest = xtest.astype('float32')

xtrain = xtrain/255 
xtest = xtest/255

batch_size = 128
classes = 10
epochs = 10

model = Sequential()
model.add(Conv2D(32, kernel_size = (5,5), activation = 'relu', input_shape = input_shape))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.4))
model.add(Dense(classes, activation = 'softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'Adadelta', metrics = ['acc'])

history = model.fit(xtrain,ytrain, batch_size = batch_size,
                   epochs = epochs, verbose = 1, validation_data = (xtest,ytest))
print('MODEL SUCCESSFULLY TRAINED')


score = model.evaluate(xtest,ytest,verbose = 0)
print('loss ',score[0])
print('accuracy ',score[1])

model.save('mnist.h5')



