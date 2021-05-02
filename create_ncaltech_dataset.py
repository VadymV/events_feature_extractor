"""
Copyright (c) 2021 Vadym Gryshchuk (vadym.gryshchuk@protonmail.com).
The MIT License (MIT)

The convertion of the N-Caltech256 dataset to the folder structure
that can be understood by torch.utils.data.DataLoader.
"""

import os
import random
import h5py
import numpy as np

# Change to select the other dataset
DATASET = "N-Caltech256"  # N-Caltech12

HDF5_INPUT_FILE = "./data/INI_Caltech256_10fps_20160424.hdf5"
OUTPUT_FOLDER = "./data/"
TRAIN_FOLDER_LL = "training"
TEST_FOLDER_LL = "testing"
OUTPUT_FOLDER = OUTPUT_FOLDER + DATASET + "/"


def write_file(data, filename, output_folder):
    out_file = os.path.join(output_folder, str(filename))

    if not os.path.exists(os.path.dirname(out_file)):
        os.makedirs(os.path.dirname(out_file))
    np.save(out_file, data)


def flip_events_along_y(events, resolution=(180, 240)):
    H, W = resolution

    events[:, 1] = H - 1 - events[:, 1]
    return events


def split_data(hdf5_input_file, output_folder, dataset):
    assert dataset in ["N-Caltech256", "N-Caltech12"]
    if dataset == "N-Caltech256":
        n_train_samples = None
        n_test_sampes = 15
    elif dataset == "N-Caltech12":
        n_train_samples = 160
        n_test_sampes = 40
    else:
        raise ValueError("dataset is not recognized")
    f = h5py.File(hdf5_input_file, 'r')
    obj_labels = list(f.keys())
    random.seed(8932857495889437)
    for obj_label in obj_labels:  # e.g. '003.backpack'
        if "101" in obj_label and dataset == "N-Caltech12":
            continue
        samples_names = list(f[obj_label].keys())
        if dataset == "N-Caltech256":
            n_train_samples = len(samples_names) - n_test_sampes
        if len(samples_names) < n_train_samples + n_test_sampes:
            continue
        print("Processing the object {}".format(obj_label))
        random.seed(4534324654154541)
        test_samples = random.sample(samples_names, n_test_sampes)
        train_samples = random.sample([x for x in samples_names if x not in test_samples], n_train_samples)
        all_samples = test_samples + train_samples
        for sample in all_samples:  # e.g. '003_0001'
            length = f[obj_label][sample]['pol'].shape[0]
            pol = f[obj_label][sample]['pol'][:length].astype(int)
            pol = np.where(pol == 0, -1, pol)
            timestamps = f[obj_label][sample]['timestamps'][:length]
            if dataset == "N-Caltech256":
                timestamps = timestamps / 1e6  # Timestamp is not used when events are converted to a histogram representation (this is done to reduce the size of data)
                data_type = np.float16
            else:
                data_type = np.float32
            x_pos = f[obj_label][sample]['x_pos'][:length]
            y_pos = f[obj_label][sample]['y_pos'][:length]
            data = np.array([x_pos, y_pos, timestamps, pol]).transpose()
            data = data.astype(data_type)
            data = flip_events_along_y(data)
            if sample in test_samples:
                test_path = os.path.join(os.path.abspath(output_folder), TEST_FOLDER_LL)
                write_file(data, sample, os.path.join(os.path.abspath(test_path), obj_label.split(".")[-1]))
            elif sample in train_samples:
                train_path = os.path.join(os.path.abspath(output_folder), TRAIN_FOLDER_LL)
                write_file(data, sample, os.path.join(os.path.abspath(train_path), obj_label.split(".")[-1]))
            else:
                raise ValueError("Logic for selecting test, validation and train data is wrong.")


split_data(HDF5_INPUT_FILE, OUTPUT_FOLDER, DATASET)
