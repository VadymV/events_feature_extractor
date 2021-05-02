"""
2021 Vadym Gryshchuk (vadym.gryshchuk@protonmail.com).

Create the numpy array and saves it to a file. The numpy array contains the data from Later it can .
"""

import os
import numpy as np

# Set values for the following two variables:
INPUT_FOLDER = "./data/extracted_features/ncaltech12/testing/"
OUTPUT_FILE = "./data/ncaltech12_ssl_features_test_data_histogram.npy"

#INPUT_FOLDER = "./data/extracted_features/voxel/ncaltech12/testing/"
#OUTPUT_FILE = "./data/ncaltech12_ssl_features_test_data_voxel.npy"


def write_file(data):

    if not os.path.exists(os.path.dirname(OUTPUT_FILE)):
        os.makedirs(os.path.dirname(OUTPUT_FILE))
    np.save(OUTPUT_FILE, data)


def create_data_for_tsne():
    features = []
    labels = []
    for _, dirs, _ in os.walk(INPUT_FOLDER):
        for dir in dirs:
            for root, _, files in os.walk(os.path.join(INPUT_FOLDER, dir)):
                for file_name in files:
                    sample = np.load(os.path.join(root, file_name))
                    features.append(sample)
                    labels.append(int(dir))
    x = np.vstack(features)
    y = np.vstack(labels)

    data = np.hstack((x, y))
    write_file(data)


create_data_for_tsne()
