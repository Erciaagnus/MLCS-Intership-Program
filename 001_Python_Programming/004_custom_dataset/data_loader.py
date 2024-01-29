#!/usr/bin/env python3
import os
import glob
import csv
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision
from torchvision import transforms


class CustomDataLoader():
    # Initialization
    def __init__(self, path='.', dataset_name='dataset', test=False, shuffle=True, normalize=True):
        # Initialize variables
        # array for each ~'s various features
        self.train_normal = []
        self.train_glioma = []
        self.train_meningioma = []
        self.train_pituitary = []
        self.test_normal = []
        self.test_glioma = []
        self.test_meningioma = []
        self.test_pituitary = []
        self.val_glioma = []
        self.val_normal = []
        self.val_meningioma = []
        self.val_pituitary = []
        # array for each ~'s label
        # Settings
        self.normalize_flag = normalize
        self.dataset_name = dataset_name
        self.shuffle = shuffle
        self.path = path
        self.glioma_img = glob.glob(os.path.join(path, 'glioma_tumor', '*.jpg'))
        self.normal_img = glob.glob(os.path.join(path, 'no_tumor', '*.jpg'))
        self.meningioma_img = glob.glob(os.path.join(path, 'meningioma_tumor', '*.jpg'))
        self.pituitary_img = glob.glob(os.path.join(path, 'pituitary_tumor', '*.jpg'))

        # Initialize labels
        self.train_normal_label = []
        self.train_glioma_label = []
        self.train_meningioma_label = []
        self.train_pituitary_label = []

    # Define normalize function
    def normalize(self, flag, arr):
        if self.normalize_flag:
            arr = (arr - arr.mean()) / (arr.std() + 1e-8)
        return arr

    # Create
    # Import all the raw datasets and convert the datasets into numpy arrays.
    def create(self):
        # 'Create' imports all the raw dataset and convert the dataset into normalized numpy arrays.
        # This code imports multiple csv files
        for img_path in self.glioma_img:
            img = Image.open(img_path)
            img_array = np.array(img)
            img_array = self.normalize(self.normalize_flag, img_array)
            self.train_glioma.append(img_array)
            self.train_glioma_label.append(1)
        for img_path in self.normal_img:
            img = Image.open(img_path)
            img_array = np.array(img)
            img_array = self.normalize(self.normalize_flag, img_array)
            self.train_normal.append(img_array)
            self.train_normal_label.append(0)
        for img_path in self.meningioma_img:
            img = Image.open(img_path)
            img_array = np.array(img)
            img_array = self.normalize(self.normalize_flag, img_array)
            self.train_meningioma.append(img_array)
            self.train_meningioma_label.append(2)
        for img_path in self.pituitary_img:
            img = Image.open(img_path)
            img_array = np.array(img)
            img_array = self.normalize(self.normalize_flag, img_array)
            self.train_pituitary.append(img_array)
            self.train_pituitary_label.append(3)

        if self.shuffle:
            random.shuffle(self.train_glioma)
            random.shuffle(self.train_normal)
            random.shuffle(self.train_meningioma)
            random.shuffle(self.train_pituitary)

        # Assuming you want to save the entire dataset, not just 'body_acc'
        np.savez(os.path.join('./datasets', self.dataset_name + '.npz'),
                 train_glioma=self.train_glioma, train_normal=self.train_normal,
                 train_meningioma=self.train_meningioma, train_pituitary=self.train_pituitary,
                 train_glioma_label=self.train_glioma_label, train_normal_label=self.train_normal_label,
                 train_meningioma_label=self.train_meningioma_label, train_pituitary_label=self.train_pituitary_label)

    def read(self):
        # Convert dataset into windowed data.
        # Not necessary if you are not dealing with sequential data or model
        dataset_path = os.path.join('./datasets', self.dataset_name + '.npz')
        if os.path.exists(dataset_path):
            dataset = np.load(dataset_path)
            self.train_glioma = dataset['train_glioma']
            self.train_normal = dataset['train_normal']
            self.train_meningioma = dataset['train_meningioma']
            self.train_pituitary = dataset['train_pituitary']
            self.train_glioma_label = dataset['train_glioma_label']
            self.train_normal_label = dataset['train_normal_label']
            self.train_meningioma_label = dataset['train_meningioma_label']
            self.train_pituitary_label = dataset['train_pituitary_label']
        else:
            print(f"Error: Dataset file {dataset_path} not found.")
    def split(self, ratio_1, ratio_2):
        # Image data
        total_data = np.concatenate([self.train_glioma, self.train_normal, self.train_meningioma, self.train_pituitary], axis=0)
        total_labels = np.concatenate([self.train_glioma_label, self.train_normal_label,
                                       self.train_meningioma_label, self.train_pituitary_label])

        if self.shuffle:
            indices = np.arange(len(total_data))
            np.random.shuffle(indices)
            total_data = total_data[indices]
            total_labels = total_labels[indices]
       # Split
        train_end = int(len(total_data) * ratio_1)
        valid_end = int(len(total_data) * (ratio_1 + ratio_2))

        train_data, train_labels = total_data[:train_end], total_labels[:train_end]
        valid_data, valid_labels = total_data[train_end:valid_end], total_labels[train_end:valid_end]
        test_data, test_labels = total_data[valid_end:], total_labels[valid_end:]

        # Save
        save_train = os.path.join(self.path, 'train.npz')
        save_valid = os.path.join(self.path, 'valid.npz')
        save_test = os.path.join(self.path, 'test.npz')

        np.savez(save_train, data=train_data, label=train_labels)
        np.savez(save_valid, data=valid_data, label=valid_labels)
        np.savez(save_test, data=test_data, label=test_labels)

if __name__ == "__main__":
    dl = CustomDataLoader()
    dl.create()
    dl.read()
    dl.split(ratio_1=0.7, ratio_2=0.15)
