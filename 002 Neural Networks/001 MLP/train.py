#!/usr/bin/env python3
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.constraints import unit_norm
from keras.optimizers import RMSprop
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# import necessary modules to define and train neural networks
batch_size = 128
num_classes= 10
epochs1 =20
epochs2=15

# the data, split between train and test sets train:test=6:1
# load mnist data file from mnist module
(x_train, y_train), (x_test, y_test)=mnist.load_data()
data_x=np.concatenate((x_train, x_test))
data_y=np.concatenate((y_train, y_test))

test_size=0.15
valid_size=0.15
# normalize: divide maximum pixel value 255
total_number=70000
train_number=int(total_number*0.7)
valid_number=int(total_number*0.15)
test_number=int(total_number*0.15)
x_train, x_test, y_train, y_test=train_test_split(data_x,data_y, test_size=test_size, random_state=1)
x_train, x_valid, y_train, y_valid=train_test_split(x_train, y_train, test_size=valid_size/(1-test_size), random_state=1)
x_train = x_train.reshape(x_train.shape[0], 784).astype('float32') /255
x_valid=x_valid.reshape(x_valid.shape[0], 784).astype('float32') / 255
x_test =x_test.reshape(x_test.shape[0], 784).astype('float32')/255


print(x_test.shape[0], 'train samples')
print(x_valid.shape[0], 'valid samples')
print(x_test.shape[0], 'test samples')

#convert class vectors to binary class matrices
y_train=keras.utils.to_categorical(y_train, num_classes)
y_valid=keras.utils.to_categorical(y_valid, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

############################################################
# Model 1
def design_model_1():

    model=Sequential()
    # Add new layers(first) to model Dense layer(fully connected layer, every input neurons was connected to previous input neurons)
    # (number of neurons(number of characteristics), activiation function(ReLU))
    model.add(Dense(512, activation='tanh', input_shape=(784,)))
    model.add(Dropout(0.2))
    model.add(Dense(500, activation='tanh'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs2, verbose=1, validation_data=(x_valid, y_valid))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    model.save('model1.keras')
    return model

# Model 2 (More Layers)
def design_model_2():
    model=Sequential()
    model.add(Dense(660, activation='relu', input_shape=(784,)))
    model.add(Dropout(0.3))
    model.add(Dense(680, activation='relu', kernel_constraint=unit_norm()))
    model.add(Dropout(0.3))
    model.add(Dense(700, activation='relu', kernel_constraint=unit_norm()))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs2, verbose=1, validation_data=(x_valid, y_valid))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    model.save('model2.keras')
    return model

model_1=design_model_1()
history_1=model_1.fit(x_train,y_train, batch_size=batch_size, epochs=epochs1, verbose=1, validation_data=(x_valid, y_valid))

model_2=design_model_2()
history_2=model_2.fit(x_train,y_train, batch_size=batch_size, epochs=epochs2, verbose=1, validation_data=(x_valid, y_valid))

