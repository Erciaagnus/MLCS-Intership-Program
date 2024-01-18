#!/usr/bin/env python3
import os
import glob
import numpy as np

# Step 1: Read the converted 2D dataset files in 'outputs' directory
path_to_train_2d_datasets = os.path.join('outputs', 'train', '*.csv')
path_to_test_2d_datasets = os.path.join('outputs', 'test', '*.csv')
train_2d_files = glob.glob(path_to_train_2d_datasets)
test_2d_files=glob.glob(path_to_test_2d_datasets)

# We already know that the width and height of the 2D MNIST dataset is 28.
train = np.empty((len(train_2d_files), 28, 28))
train_labels = np.empty((len(train_2d_files), 1), dtype=int)

# Step 2: Create empty 3D numpy arrays to save loaded 2D MNIST data
for data_idx, data_path in enumerate(train_2d_files):

# Step 3: Read the csv and replace the elements
    csv_data = np.loadtxt(data_path, delimiter=",")
    for i in range(28):
        for j in range(28):
            train[data_idx, i, j] = csv_data[i, j]

# Step 4: Create additional 1D numpy arrays to save the label of the loaded 2D MNIST data


for data_idx, data_path in enumerate(train_2d_files):
    # Extract label from the file name, assuming the format is 'label-index.csv'
    label = int(os.path.basename(data_path).split("-")[0])
    train_labels[data_idx] = label

# Step 5: Concatenate the train arrays into one array
expanded_labels = np.expand_dims(train_labels, axis=1)
expanded_labels = np.repeat(expanded_labels, 28, axis =1)
expanded_labels = np.repeat(expanded_labels, 28, axis=2)
train_data = np.concatenate((train, expanded_labels), axis=2)

# Step 6: Shuffle the numpy array
np.random.shuffle(train_data)

# Step 7: Split the numpy array into train, validation, and test arrays
split_ratio = [0.7, 0.2, 0.1]
total_samples = len(train_data)
train_split = int(split_ratio[0] * total_samples)
valid_split = train_split + int(split_ratio[1] * total_samples)

train_set = train_data[:train_split, :, :]
valid_set = train_data[train_split:valid_split, :, :]
test_set = train_data[valid_split:, :, :]

# Step 8: Save the splitted arrays into .npz files
path_to_save = os.path.join('outputs', 'mnist_dataset.npz')
np.savez(
    path_to_save,
    train_x=train_set[:, :, :-1],
    train_y=train_set[:, 0, -1],
    valid_x=valid_set[:, :, :-1],
    valid_y=valid_set[:, 0, -1],
    test_x=test_set[:, :, :-1],
    test_y=test_set[:, 0, -1],
)

print("Dataset creation and saving completed.")
