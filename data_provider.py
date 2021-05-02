"""
This file is created by Vadym Gryshchuk (vadym.gryshchuk@protonmail.com).

Some parts of the code (adapted) in this file are taken from different sources. See methods and classes.

The data provider for the N-MNIST and N-Caltech datasets.
"""

import random
from os import listdir
from os.path import join

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.backends import cudnn
from torch.cuda import manual_seed, manual_seed_all
from torch.utils import data as torch_data


def random_shift_events(events, max_shift=20, resolution=(180, 240)):
    """
    Shift events.
    The code (adapted) in this method is taken
    from https://github.com/uzh-rpg/rpg_asynet/blob/master/dataloader/dataset.py.
    """

    H, W = resolution

    x_shift, y_shift = np.random.randint(-max_shift, max_shift + 1, size=(2,))

    events[:, 0] += x_shift
    events[:, 1] += y_shift

    valid_events = (events[:, 0] >= 0) & (events[:, 0] < W) & (events[:, 1] >= 0) & (events[:, 1] < H)
    events = events[valid_events]

    return events


def random_flip_events_along_x(events, resolution=(180, 240), p=0.5):
    """
    Flip events along x-axis.
    The code (adapted) in this method is taken
    from https://github.com/uzh-rpg/rpg_asynet/blob/master/dataloader/dataset.py.
    """
    H, W = resolution
    if np.random.random() < p:
        events[:, 0] = W - 1 - events[:, 0]

    return events


def flip_events_along_y(events, resolution=(180, 240)):
    """
    Flip events along y-axis.
    The code in this method is based on https://github.com/uzh-rpg/rpg_asynet/blob/master/dataloader/dataset.py.
    """
    H, W = resolution
    events[:, 1] = H - 1 - events[:, 1]
    return events


def random_swap_channels(histogram, p=0.2):
    if np.random.random() < p:
        channel_zero = histogram[..., 0].copy()
        channel_one = histogram[..., 1].copy()
        histogram[..., 0] = channel_one
        histogram[..., 1] = channel_zero

    return histogram


def random_change_brightness(histogram, p=0.2):
    if np.random.random() < p:
        mask_zero = torch.randint(-1, 3, (histogram.shape[0], histogram.shape[1])).numpy()
        mask_one = torch.randint(-1, 3, (histogram.shape[0], histogram.shape[1])).numpy()

        histogram[:, :, 0] = np.where(histogram[:, :, 0] > 1, histogram[:, :, 0] + mask_zero, histogram[:, :, 0])
        histogram[:, :, 1] = np.where(histogram[:, :, 1] > 1, histogram[:, :, 1] + mask_one, histogram[:, :, 1])

    return histogram


def generate_event_histogram(events, shape):
    """
    Generates a histogram from events.
    The code in this method is taken from https://github.com/uzh-rpg/rpg_asynet/blob/master/dataloader/dataset.py.
    """
    H, W = shape
    x, y, t, p = events.T
    x = x.astype(np.int)
    y = y.astype(np.int)

    img_pos = np.zeros((H * W,), dtype="float32")
    img_neg = np.zeros((H * W,), dtype="float32")

    np.add.at(img_pos, x[p == 1] + W * y[p == 1], 1)
    np.add.at(img_neg, x[p == -1] + W * y[p == -1], 1)

    histogram = np.stack([img_neg, img_pos], -1).reshape((H, W, 2))

    return histogram


def events_to_voxel_grid(events, num_bins, width, height):
    """
    The code in this method is taken
    from https://github.com/uzh-rpg/rpg_e2vid/blob/d0a7c005f460f2422f2a4bf605f70820ea7a1e5f/utils/inference_utils.py
    Creates a voxel grid.
    """

    assert (events.shape[1] == 4)
    assert (num_bins > 0)
    assert (width > 0)
    assert (height > 0)

    voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()

    # normalize the event timestamps so that they lie between 0 and num_bins
    last_stamp = events[-1, 2]
    first_stamp = events[0, 2]
    deltaT = last_stamp - first_stamp

    if deltaT == 0:
        deltaT = 1.0

    events[:, 2] = (num_bins - 1) * (events[:, 2] - first_stamp) / deltaT
    ts = events[:, 2]
    xs = events[:, 0].astype(np.int)
    ys = events[:, 1].astype(np.int)
    pols = events[:, 3]
    pols[pols == 0] = -1  # polarity should be +1 / -1

    tis = ts.astype(np.int)
    dts = ts - tis
    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts

    valid_indices = tis < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
              + tis[valid_indices] * width * height, vals_left[valid_indices])

    valid_indices = (tis + 1) < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
              + (tis[valid_indices] + 1) * width * height, vals_right[valid_indices])

    voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))

    return voxel_grid


class NCaltech:
    """
    The structure of the code in this class is inspired
    by https://github.com/uzh-rpg/rpg_asynet/blob/master/dataloader/dataset.py

    Some parts of the code in this class are copied
    from https://github.com/uzh-rpg/rpg_asynet/blob/master/dataloader/dataset.py
    """

    def __init__(self, repeat_augmentations, root, height=180, width=240, nr_events_window=0, augmentation=False,
                 shuffle=True, transform=None, events_representation=None):

        self.object_classes = listdir(root)
        self.repeat_augmentations = repeat_augmentations
        self.transform = transform
        self.width = width
        self.height = height
        self.augmentation = augmentation
        self.nr_events_window = nr_events_window
        self.nr_classes = len(self.object_classes)
        self.events_mode = events_representation

        self.files = []
        self.labels = []

        for i, object_class in enumerate(self.object_classes):
            new_files = [join(root, object_class, f) for f in listdir(join(root, object_class))]
            self.files += new_files
            self.labels += [i] * len(new_files)

        self.nr_samples = len(self.labels)

        if shuffle:
            zipped_lists = list(zip(self.files, self.labels))
            random.seed(7)
            random.shuffle(zipped_lists)
            self.files, self.labels = zip(*zipped_lists)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        returns events and label, loading events from aedat
        :param idx:
        :return: x,y,t,p,  label
        """
        label = self.labels[idx]
        filename = self.files[idx]
        orig_events = np.load(filename).astype(np.float32)

        nr_events = orig_events.shape[0]

        if self.repeat_augmentations is None:
            self.repeat_augmentations = 1

        event_representation_list = list()
        assert self.repeat_augmentations >= 1
        for _ in range(self.repeat_augmentations):
            window_start = 0
            window_end = nr_events
            events = orig_events.copy()
            if self.augmentation:
                events = random_shift_events(events)
                events = random_flip_events_along_x(events)
            if self.nr_events_window != 0:
                window_start = random.randrange(0, max(1, nr_events - self.nr_events_window))
                # Catch case if number of events in batch is lower than number of events in window:
                window_end = min(nr_events, window_start + self.nr_events_window)

            events = events[window_start:window_end, :]

            if self.events_mode is None:
                random.seed(1243253454234)
                indices = random.sample(range(0, nr_events), max(2000, int(nr_events * 0.01)))
                indices.sort()
                events = events[indices]
                if events[:, 2][-1] < 10:
                    events[:, 2] = events[:, 2] * 1e6
                events[:, 3] = np.where(events[:, 3] == -1, 0, events[:, 3])
                events = torch.from_numpy(events)
                return events, None, label

            if self.events_mode == "voxel":
                event_representation = events_to_voxel_grid(events, num_bins=10, width=self.width, height=self.height)
                event_representation = np.moveaxis(event_representation, 0, -1)
            else:
                event_representation = generate_event_histogram(events, (self.height, self.width))

            if self.transform is not None:
                event_representation_swaped_axis = np.moveaxis(event_representation, -1, 0)
                event_representation_transformed = self.transform(
                    torch.from_numpy(event_representation_swaped_axis)).numpy()
                event_representation_transformed = np.moveaxis(event_representation_transformed, 0, -1)
                event_representation = event_representation_transformed

            event_representation_list.append(event_representation)

        return event_representation, event_representation_list, label


class DataProvider():
    def __init__(self, seed):
        torch.manual_seed(seed)
        manual_seed(seed)
        manual_seed_all(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        random.seed(seed)
        np.random.seed(seed)
        self.nr_train_epochs = None

    @staticmethod
    def get_num_classes(dataset):
        if dataset == "ncaltech256":
            return 257
        elif dataset == "nmnist":
            return 10
        elif dataset == "ncaltech12":
            return 12
        elif dataset == "ncaltech101":
            return 101

    @staticmethod
    def get_train_transforms(method, dataset):
        if "ncaltech" in dataset:
            side = (180, 240)
        elif dataset == "nmnist":
            side = (34, 34)
        else:
            raise ValueError("Check code")

        if method == "simclr" and "ncaltech" in dataset:
            rnd_hflip = transforms.RandomHorizontalFlip(p=0.5)
            rnd_resizedcrop = transforms.RandomResizedCrop(size=side, scale=(0.08, 1.0),
                                                           ratio=(0.75, 1.3333333333333333),
                                                           interpolation=transforms.InterpolationMode.BILINEAR)
            train_transform = transforms.Compose([rnd_resizedcrop, rnd_hflip])
        elif method == "simclr" and dataset == "nmnist":
            rnd_rot = transforms.RandomRotation(10., interpolation=transforms.InterpolationMode.BILINEAR)
            rnd_hflip = transforms.RandomHorizontalFlip(p=0.5)
            rnd_erase = transforms.RandomErasing()
            rnd_blur = transforms.GaussianBlur(
                (int(side[1] * .1) if int(side[1] * .1) % 2 != 0 else int(side[1] * .1) + 1),
                int(side[0] * .1))
            rnd_resizedcrop = transforms.RandomResizedCrop(size=side, scale=(0.08, 1.0),
                                                           ratio=(0.75, 1.3333333333333333),
                                                           interpolation=transforms.InterpolationMode.BILINEAR)
            train_transform = transforms.Compose([rnd_resizedcrop, rnd_hflip, rnd_blur, rnd_rot, rnd_erase])
        elif method == "classifier" and ("ncaltech" in dataset or dataset == "nmnist"):
            train_transform = None
        elif method == "linear_evaluation" and ("ncaltech" in dataset or dataset == "nmnist"):
            train_transform = None
        else:
            raise ValueError("The method {} is not recognized".format(str(method)))
        return train_transform

    def get_train_loader(self, dataset, data_type, data_size, train_transform, repeat_augmentations, num_workers,
                         drop_last, nr_events_window, events_representation):
        if data_type == "simclr":
            if dataset == "ncaltech256":
                train_set = NCaltech(repeat_augmentations, root="./data/N-Caltech256/training",
                                     transform=train_transform, augmentation=False, nr_events_window=nr_events_window,
                                     events_representation=events_representation)
            elif dataset == "ncaltech12":
                train_set = NCaltech(repeat_augmentations, root="./data/N-Caltech12/training",
                                     transform=train_transform,
                                     augmentation=False, nr_events_window=nr_events_window,
                                     events_representation=events_representation)
            elif dataset == "nmnist":
                train_set = NMNIST(repeat_augmentations, root="./data/N-MNIST/training",
                                   transform=train_transform,
                                   augmentation=False, nr_events_window=nr_events_window,
                                   events_representation=events_representation)
            elif dataset == "ncaltech101":
                train_set = NCaltech(repeat_augmentations, root="./data/N-Caltech101/training",
                                     transform=train_transform,
                                     augmentation=False, nr_events_window=nr_events_window,
                                     events_representation=events_representation)
            else:
                raise ValueError("Check code")
            train_loader = torch_data.DataLoader(train_set, batch_size=data_size, shuffle=True,
                                                 num_workers=num_workers, pin_memory=True, drop_last=drop_last)
        elif data_type == "classifier":
            if dataset == "ncaltech256":
                train_set = NCaltech(repeat_augmentations, root="./data/N-Caltech256/training",
                                     transform=train_transform,
                                     augmentation=True, nr_events_window=nr_events_window,
                                     events_representation=events_representation)
            elif dataset == "ncaltech12":
                train_set = NCaltech(repeat_augmentations, root="./data/N-Caltech12/training",
                                     transform=train_transform,
                                     augmentation=True, nr_events_window=nr_events_window,
                                     events_representation=events_representation)
            elif dataset == "nmnist":
                train_set = NMNIST(repeat_augmentations, root="./data/N-MNIST/training",
                                   transform=train_transform,
                                   augmentation=False, nr_events_window=nr_events_window,
                                   events_representation=events_representation)
            elif dataset == "ncaltech101":
                train_set = NCaltech(repeat_augmentations, root="./data/N-Caltech101/training",
                                     transform=train_transform,
                                     augmentation=True, nr_events_window=nr_events_window,
                                     events_representation=events_representation)
            else:
                raise ValueError("Check code")
            train_loader = torch_data.DataLoader(train_set, batch_size=data_size, shuffle=True,
                                                 num_workers=num_workers, pin_memory=True, drop_last=drop_last)
        elif data_type == "raw":
            if dataset == "nmnist":
                train_set = NMNIST(repeat_augmentations, root="./data/N-MNIST/training",
                                   nr_events_window=nr_events_window, events_representation=None)
            elif dataset == "ncaltech256":
                train_set = NCaltech(repeat_augmentations, root="./data/N-Caltech256/training",
                                     nr_events_window=nr_events_window, events_representation=None)
            elif dataset == "ncaltech12":
                train_set = NCaltech(repeat_augmentations, root="./data/N-Caltech12/training",
                                     nr_events_window=nr_events_window, events_representation=None)
            elif dataset == "ncaltech101":
                train_set = NCaltech(repeat_augmentations, root="./data/N-Caltech101/training",
                                     nr_events_window=nr_events_window, events_representation=None)
            else:
                raise ValueError("Check code")

            train_loader = torch_data.DataLoader(train_set, batch_size=data_size,
                                                 num_workers=num_workers, pin_memory=True,
                                                 collate_fn=collate_events, drop_last=drop_last,
                                                 shuffle=True)
        elif data_type == "extract":
            if dataset == "nmnist":
                train_set = NMNIST(repeat_augmentations, root="./data/N-MNIST/training",
                                   nr_events_window=nr_events_window, events_representation=events_representation)
            elif dataset == "ncaltech256":
                train_set = NCaltech(repeat_augmentations, root="./data/N-Caltech256/training",
                                     nr_events_window=nr_events_window, events_representation=events_representation)
            elif dataset == "ncaltech12":
                train_set = NCaltech(repeat_augmentations, root="./data/N-Caltech12/training",
                                     nr_events_window=nr_events_window, events_representation=events_representation)
            elif dataset == "ncaltech101":
                train_set = NCaltech(repeat_augmentations, root="./data/N-Caltech101/training",
                                     nr_events_window=nr_events_window, events_representation=events_representation)
            else:
                raise ValueError("Check code")

            train_loader = torch_data.DataLoader(train_set,
                                                 batch_size=data_size,
                                                 num_workers=num_workers,
                                                 pin_memory=True,
                                                 drop_last=drop_last,
                                                 shuffle=True)
        else:
            raise ValueError("The type {} is not recognizes".format(str(data_type)))

        self.nr_train_epochs = int(train_set.nr_samples / data_size) + 1
        return train_loader, train_set

    def get_test_loader(self, dataset, data_size, num_workers, nr_events_window, events_representation,
                        collate_fn=None):
        if dataset == "ncaltech256":
            test_set = NCaltech(repeat_augmentations=None, root="./data/N-Caltech256/testing",
                                transform=None, augmentation=False, nr_events_window=nr_events_window,
                                events_representation=events_representation)
        elif dataset == "ncaltech12":
            test_set = NCaltech(repeat_augmentations=None, root="./data/N-Caltech12/testing",
                                transform=None, augmentation=False, nr_events_window=nr_events_window,
                                events_representation=events_representation)
        elif dataset == "nmnist":
            test_set = NMNIST(repeat_augmentations=None, root="./data/N-MNIST/testing",
                              transform=None, augmentation=False, nr_events_window=nr_events_window,
                              events_representation=events_representation)
        elif dataset == "ncaltech101":
            test_set = NCaltech(repeat_augmentations=None, root="./data/N-Caltech101/testing",
                                transform=None, augmentation=False, nr_events_window=nr_events_window,
                                events_representation=events_representation)
        else:
            raise ValueError("Check code")
        test_loader = torch_data.DataLoader(test_set, batch_size=data_size, shuffle=False,
                                            num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)
        return test_loader


class NMNIST:
    """
    The structure of the code in this class is inspired
    by https://github.com/uzh-rpg/rpg_asynet/blob/master/dataloader/dataset.py

    Some parts of the code in this class are copied
    from https://github.com/uzh-rpg/rpg_asynet/blob/master/dataloader/dataset.py
    """

    def __init__(self, repeat_augmentations, root, height=34, width=34, nr_events_window=0, augmentation=False,
                 shuffle=True, transform=None, events_representation=None):

        self.object_classes = listdir(root)
        self.repeat_augmentations = repeat_augmentations
        self.transform = transform
        self.width = width
        self.height = height
        self.augmentation = augmentation
        self.nr_events_window = nr_events_window
        self.nr_classes = len(self.object_classes)
        self.events_mode = events_representation

        self.files = []
        self.labels = []

        for i, object_class in enumerate(self.object_classes):
            new_files = [join(root, object_class, f) for f in listdir(join(root, object_class))]
            self.files += new_files
            self.labels += [i] * len(new_files)

        self.nr_samples = len(self.labels)

        if shuffle:
            zipped_lists = list(zip(self.files, self.labels))
            random.seed(7)
            random.shuffle(zipped_lists)
            self.files, self.labels = zip(*zipped_lists)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        label = self.labels[idx]
        filename = self.files[idx]
        orig_events = self.convert_binary_file(filename)
        nr_events = orig_events.shape[0]

        if self.repeat_augmentations is None:
            self.repeat_augmentations = 1

        histograms_list = list()
        assert self.repeat_augmentations >= 1
        for _ in range(self.repeat_augmentations):
            window_start = 0
            window_end = nr_events
            events = orig_events.copy()
            if self.nr_events_window != 0:
                window_start = random.randrange(0, max(1, nr_events - self.nr_events_window))
                # Catch case if number of events in batch is lower than number of events in window:
                window_end = min(nr_events, window_start + self.nr_events_window)

            events = events[window_start:window_end, :]

            if self.events_mode is None:
                random.seed(1243253454234)
                indices = random.sample(range(0, nr_events), min(2000, nr_events))
                indices.sort()
                events = events[indices]
                events[:, 3] = np.where(events[:, 3] == -1, 0, events[:, 3])
                events = torch.from_numpy(events)
                return events, None, label

            if self.events_mode == "voxel":
                event_representation = events_to_voxel_grid(events, num_bins=10, width=self.width, height=self.height)
                event_representation = np.moveaxis(event_representation, 0, -1)
            else:
                event_representation = generate_event_histogram(events, (self.height, self.width))

            if self.transform is not None:
                histogram_swaped_axis = np.moveaxis(event_representation, -1, 0)
                histogram_transformed = self.transform(torch.from_numpy(histogram_swaped_axis)).numpy()
                histogram_transformed = np.moveaxis(histogram_transformed, 0, -1)
                event_representation = histogram_transformed
                event_representation = random_swap_channels(event_representation)
                event_representation = random_change_brightness(event_representation)

            histograms_list.append(event_representation)

        return event_representation, histograms_list, label

    @staticmethod
    def convert_binary_file(filename):
        """
        The code is taken from https://github.com/gorchard/event-Python/blob/master/eventvision.py
        The code is adapted.
        """
        f = open(filename, 'rb')
        raw_data = np.fromfile(f, dtype=np.uint8)
        f.close()
        raw_data = np.int32(raw_data)

        all_y = raw_data[1::5]
        all_x = raw_data[0::5]
        all_p = (raw_data[2::5] & 128) >> 7  # bit 7
        all_ts = ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])

        # Process time stamp overflow events:
        time_increment = 2 ** 13
        overflow_indices = np.where(all_y == 240)[0]
        for overflow_index in overflow_indices:
            all_ts[overflow_index:] += time_increment

        # Everything else is a proper td spike:
        td_indices = np.where(all_y != 240)[0]

        # Replace zeros with -1:
        all_p = np.where(all_p == 0, -1, all_p)

        data = np.array([all_x[td_indices], all_y[td_indices], all_ts[td_indices], all_p[td_indices]]).transpose()
        data = data.astype(np.float32)

        return data


def _pad_sequences(sequences, sequence_starts, padding_value=0.0):
    """
    Input sequences can be of different lengths (dimension seq_length). These sequences must be padded in order to put
    more than one sample into a batch of data. Therefore, the size of the seq_length dimension of a tensor
    that represents a batch of data depends on the length of the largest sequence in (sequences).
    Padding makes sense when it is done at the very beginning. Consequently, one needs to find when a sequence starts
    and pad everything before with zeros.
    :param sequences: a tuple of sequences of size batch_size.
        A sequence is a tensor of shape (seq_length, features+timestamp).
    :param sequence_starts: a numpy array of shape (batch_size), where each element indicates when a sequence
        should start.
    :param padding_value: a padding value.
    :return: padded sequences.
    """
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max([s.size(0) for s in sequences])
    out_dims = (len(sequences), max_len) + trailing_dims

    out_tensor = sequences[0].new_full(out_dims, padding_value)
    for i, tensor in enumerate(sequences):
        out_tensor[i, sequence_starts[i]:, ...] = tensor

    return out_tensor


def collate_events(batch):
    """
    Bring inputs, labels and positions when each sequence starts into a batch of data together.
    :param batch: a list of tuples. Each tuple contains a tensor and an int value.
        A tensor is of shape (seq_length, features+timestamp). An int value represents a class label.
    :return: a batch of data composed of input sequences, labels and positions when each sequence starts.
    """

    (inputs, _, labels) = zip(*batch)
    labels = [x for x in labels]

    sequence_lengths = np.asarray([len(x) for x in inputs])  # lengths of sequences for each sample
    max_seq_value = max(sequence_lengths)  # maximum length of a sequence from all samples
    max_length_seq = np.full(shape=len(sequence_lengths), fill_value=max_seq_value)
    sequences_starts = np.subtract(max_length_seq, sequence_lengths)  # start positions of sequences

    padded_inputs = _pad_sequences(inputs, sequences_starts)

    return padded_inputs, torch.LongTensor(sequences_starts), torch.LongTensor(labels)
