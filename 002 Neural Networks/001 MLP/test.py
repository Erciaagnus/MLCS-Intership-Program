#!/usr/bin/env python3
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import numpy as np
import matplotlib.pyplot as plt
import train
from sklearn.model_selection import train_test_split

x_train=train.x_train
x_test=train.x_test
x_valid=train.x_valid

y_train=train.y_train
y_test=train.y_test
y_valid=train.y_valid

model_1=train.model_1
model_2=train.model_2

history_1=train.history_1
history_2=train.history_2

epochs_1=train.epochs1
epochs_2=train.epochs2
batch_size=train.batch_size

#Evaluate model_1, model_2

score_1=model_1.evaluate(x_test, y_test, verbose=0)
score_2=model_2.evaluate(x_test, y_test, verbose=0)


#plot accuracy
plt.plot(history_1.history['accuracy'])
plt.plot(history_1.history['val_accuracy'])
plt.title('model_1 accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history_2.history['accuracy'])
plt.plot(history_2.history['val_accuracy'])
plt.title('model_2 accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
print('model1 : Test loss: ', score_1[0], '  , Test accuracy: ', score_1[1])
print('model2 : Test loss: ', score_2[0], '  , Test accuracy: ', score_2[1])
