#! /usr/bin/env python3
import os
import csv
import numpy as np

script_directory = os.path.dirname(os.path.abspath(__file__))
path_to_train_dataset_1 = os.path.join(script_directory, "datasets", "mnist_train.csv")
path_to_train_dataset_2 = os.path.join(
    script_directory, "datasets", "mnist_train_2.csv"
)
path_to_train_dataset_3 = os.path.join(script_directory, "datasets", "mnist_test.csv")

output_path_train = os.path.join("outputs", "train")
if not os.path.exists(output_path_train):
    os.makedirs(output_path_train)


def convert_to_2d_vector(data):
    # Implement the logic to convert 1D vector to 2D vector here
    # For example, if each row contains 764 values for 28x28 image, reshape it to 28 x 28 array
    # Create 'outputs/train' directory if not exists

    return np.array(data).reshape((28, 28))


with open(path_to_train_dataset_1, "r") as csv_file:
    reader = csv.reader(csv_file)
    next(reader)
    for row_index, row in enumerate(reader):
        label = row[0]
        image_data = row[1:]

        # Convert 1D vector to 2D vector
        tow_d_vector = convert_to_2d_vector(image_data)

        output_filename = f"{label}-{row_index}.csv"
        output_filepath = os.path.join(output_path_train, output_filename)

        with open(output_filepath, "w", newline="") as output_csv:
            csv_writer = csv.writer(output_csv)
            csv_writer.writerows(tow_d_vector)

print("Conversion and saving completed.")
