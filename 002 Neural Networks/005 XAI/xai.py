#!/usr/bin/env python3
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import tensorflow as tf
#Load custom datasets
train_data=np.load('/004_custom_dataset/train.npz')
valid_data=np.load('/004_custom_dataset/valid.npz')
test_data=np.load('/004_custom_dataset/test.npz')

train_x=train_data['train_x']
train_y=train_data['train_y']

valid_x=valid_data['valid_x']
valid_y=valid_data['valid_y']

test_x=test_data['test_x']
test_y=test_data['test_y']


#Change the labels(str) to numeric
def labels_to_numeric(labels):
    #create a unique sorted list of all labels.
    unique_labels=sorted(set(labels))
    #Create a dictionary mapping labels to numbers
    labels_to_number={label: number for number, label in enumerate(unique_labels)}
    
    # map the labels to numbers
    numeric_labels=[labels_to_number[label] for label in labels]
    return numeric_labels
train_y=labels_to_numeric(train_y)
valid_y=labels_to_numeric(valid_y)
test_y=labels_to_numeric(test_y)
# def preprocessing
def preprocessing(image, label):
    image=tf.cast(image, tf.float32)
    image=image/255.
    image=tf.image.resize(image, (224, 224))
    return image, label

#prepare batches
train_batches=train_data.shuffle(num_examples //4).map(preprocessing).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
#Transfrom the data using tf.data.Dataset
#Apply preprocessing function
#Model, Used CNN Model VGG16 for pretrained base model
#Training
# Set the CAM model and show the heatmap of 3 example data