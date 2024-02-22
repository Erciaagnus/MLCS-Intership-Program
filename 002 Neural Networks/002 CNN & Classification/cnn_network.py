#!/usr/bin/env python3
import keras
from keras.datasets.cifar10 import load_data
import numpy as np
import tensorflow as tf
from keras.applications.xception import Xception

class ImageClassification():
    def __init__(self):
        self.model = keras.models.Sequential()  # Sequential 모델 객체 생성

    def create(self, ratio_1, ratio_2):
        (x_train, y_train), (x_test, y_test) = load_data()
        # 편리하게 나누기 위해 1차원으로 이어줌
        x_total = np.concatenate([x_train, x_test])
        y_total = np.concatenate([y_train, y_test])
        # split data
        [x_train, x_valid, x_test] = np.split(x_total, [int(60000 * ratio_1), int(60000 * (ratio_1 + ratio_2))])
        [y_train, y_valid, y_test] = np.split(y_total, [int(60000 * ratio_1), int(60000 * (ratio_1 + ratio_2))])

        # normalize
        x_train = keras.applications.xception.preprocess_input(x_train)
        y_train = keras.utils.to_categorical(y_train)
        x_valid = keras.applications.xception.preprocess_input(x_valid)
        y_valid = keras.utils.to_categorical(y_valid)
        x_test = keras.applications.xception.preprocess_input(x_test)
        y_test = keras.utils.to_categorical(y_test)

        return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)

    def build(self):
        initialize = keras.initializers.he_normal()
        input_tensor = keras.Input(shape=(32, 32, 3))
        resized_images = keras.layers.Lambda(lambda image: tf.image.resize(image, (128, 128)))(input_tensor)
        self.base_model = Xception(include_top=False, weights='imagenet', input_tensor=resized_images,
                                   input_shape=(128, 128, 3), pooling='max')
        self.base_model.trainable = False

        # Add layers to define the architecture
        self.model.add(self.base_model)
        self.model.add(keras.layers.Dense(512, activation='relu', kernel_initializer=initialize,
                                           kernel_constraint=keras.constraints.unit_norm(),
                                           kernel_regularizer=keras.regularizers.l2(1e-4)))
        self.model.add(keras.layers.Dropout(0.3))
        self.model.add(keras.layers.Dense(256, activation='relu', kernel_initializer=initialize,
                                           kernel_constraint=keras.constraints.unit_norm(),
                                           kernel_regularizer=keras.regularizers.l2(1e-4)))
        self.model.add(keras.layers.Dropout(0.3))
        self.model.add(keras.layers.Dense(10, activation='softmax'))
        self.model.summary()

        self.model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                           metrics=['accuracy'])
        return self.model

def main():
    ratio_1 = 0.7
    ratio_2 = 0.2
    cm = ImageClassification()
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = cm.create(ratio_1, ratio_2)
    model = cm.build()
    # 모델 학습, 평가 코드 추가해야 함

if __name__ == "__main__":
    main()
