#!/usr/bin/env python3
import os
import glob
import numpy as np
import random
from PIL import Image

class CustomDataLoader():
    def __init__(self, path='.', dataset_name='dataset', shuffle=True, normalize=True):
        self.path = path
        self.dataset_name = dataset_name
        self.shuffle = shuffle
        self.normalize_flag = normalize
        self.data = []
        self.labels = []

    def normalize(self, arr):
        if self.normalize_flag:
            arr = (arr - arr.mean()) / (arr.std() + 1e-8)
        return arr

    def load_images(self, folder, label):
        images = glob.glob(os.path.join(self.path, folder, '*.jpg'))
        data = []
        labels = []
        for img_path in images:
            img = Image.open(img_path)
            img_array = np.array(img)
            img_array = self.normalize(img_array)
            data.append(img_array)
            labels.append(label)
        return data, labels

    def create(self):
        glioma_data, glioma_labels = self.load_images('glioma_tumor', 1)
        normal_data, normal_labels = self.load_images('no_tumor', 0)
        meningioma_data, meningioma_labels = self.load_images('meningioma_tumor', 2)
        pituitary_data, pituitary_labels = self.load_images('pituitary_tumor', 3)

        self.data = glioma_data + normal_data + meningioma_data + pituitary_data
        self.labels = glioma_labels + normal_labels + meningioma_labels + pituitary_labels

        if self.shuffle:
            indices = np.arange(len(self.data))
            np.random.shuffle(indices)
            self.data = np.array(self.data)[indices]
            self.labels = np.array(self.labels)[indices]

        np.savez(os.path.join('./datasets', f'{self.dataset_name}.npz'),
                 data=self.data, labels=self.labels)

    def read(self):
        dataset_path = os.path.join('./datasets', f'{self.dataset_name}.npz')
        try:
            dataset = np.load(dataset_path)
            self.data = dataset['data']
            self.labels = dataset['labels']
        except FileNotFoundError:
            print(f"Error: Dataset file {dataset_path} not found.")

    def split(self, ratio_1, ratio_2):
        total_data = self.data
        total_labels = self.labels

        train_end = int(len(total_data) * ratio_1)
        valid_end = int(len(total_data) * (ratio_1 + ratio_2))

        train_data, train_labels = total_data[:train_end], total_labels[:train_end]
        valid_data, valid_labels = total_data[train_end:valid_end], total_labels[train_end:valid_end]
        test_data, test_labels = total_data[valid_end:], total_labels[valid_end:]

        save_train = os.path.join(self.path, 'train.npz')
        save_valid = os.path.join(self.path, 'valid.npz')
        save_test = os.path.join(self.path, 'test.npz')

        np.savez(save_train, data=train_data, labels=train_labels)
        np.savez(save_valid, data=valid_data, labels=valid_labels)
        np.savez(save_test, data=test_data, labels=test_labels)

if __name__ == "__main__":
    dl = CustomDataLoader()
    dl.create()
    dl.read()
    dl.split(ratio_1=0.7, ratio_2=0)
