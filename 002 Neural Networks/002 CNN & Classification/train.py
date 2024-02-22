#!/usr/bin/env python3
import keras
import datetime
import numpy as np
from cnn_network import ImageClassification

# cnn_network 모듈에서 ImageClassification 클래스 가져오기
# 데이터셋 분할 및 비율 설정

ratio_1 = 0.7
ratio_2 = 0.2

def main():
    # ImageClassification 클래스 인스턴스 생성
    cm = ImageClassification()

    # 모델 생성
    model = cm.build()

    # 데이터 로드 및 전처리
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = cm.create(ratio_1, ratio_2)

    # 사전 훈련된 base_model의 가중치를 동결
    cm.base_model.trainable = False

    # Optimizer 및 Learning Rate Scheduler 설정
    schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-4, decay_steps=10000, decay_rate=0.9)
    reduce = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.6, patience=2, verbose=1, mode='max', min_lr=1e-7)
    early_stop = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=21, verbose=1, restore_best_weights=True, mode='max')

    # 모델 체크포인트 설정
    checkpoint_path = 'checkpoint/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '/cifar10.h5'
    checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                  monitor='val_accuracy',
                                                  verbose=1,
                                                  save_weights_only=False,
                                                  save_best_only=True,
                                                  mode='max',
                                                  save_freq='epoch')

    # Tensorboard 로그 설정
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # 데이터 증강을 포함한 Generator 생성
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_generator = train_datagen.flow(x_train,
                                         y_train,
                                         batch_size=32)

    val_datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_generator = val_datagen.flow(x_valid,
                                     y_valid,
                                     batch_size=32)

    # 모델 컴파일
    model.compile(loss="categorical_crossentropy",
                  optimizer=keras.optimizers.Adam(learning_rate=1e-5),
                  metrics=['accuracy'])

    # 모델 학습
    model.fit(train_generator,
              epochs=19,
              validation_data=val_generator,
              verbose=1,
              callbacks=[checkpoint, tensorboard_callback, reduce, early_stop])

    # 모델 평가
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print('Test accuracy:', test_acc)

if __name__ == "__main__":
    main()
