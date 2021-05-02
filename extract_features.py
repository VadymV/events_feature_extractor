"""
2021 Vadym Gryshchuk (vadym.gryshchuk@protonmail.com).

The main routine for extracting features from SCNN.
"""

import argparse
import os
import random

import numpy as np
import torch.optim
from torch.backends import cudnn
from torch.cuda import manual_seed, manual_seed_all

from backbones.scnn import SCNN, SCNNParallel
from data_provider import DataProvider
from method.classifier import Classifier
from settings import Settings

parser = argparse.ArgumentParser(description='Train a feature extractor')
parser.add_argument('--settings_file', help='Path to settings yaml', required=False, default='./settings.yaml')

args = parser.parse_args()
settings_filepath = args.settings_file
settings = Settings(settings_filepath)

manual_seed(settings.seed)
manual_seed_all(settings.seed)
cudnn.deterministic = True
cudnn.benchmark = False
torch.manual_seed(settings.seed)
random.seed(settings.seed)
np.random.seed(settings.seed)

if not torch.cuda.is_available():
    print("CUDA is not available.")
print("Device: {}".format(str(settings.device)))
print("Seed: {}".format(str(settings.seed)))

feature_extractor = SCNN(input_channels=10 if settings.events_representation == "voxel" else 2)
feature_extractor = SCNNParallel(feature_extractor).to(settings.device)
print(feature_extractor)

total_parameterss = sum(param.numel() for param in feature_extractor.parameters() if param.requires_grad)
print("Total trainable parameters: " + str(total_parameterss))

data_provider = DataProvider(settings.seed)
num_classes = data_provider.get_num_classes(settings.dataset_name)

test_loader = data_provider.get_test_loader(dataset=settings.dataset_name,
                                            data_size=settings.batch_size,
                                            num_workers=settings.num_workers,
                                            events_representation=settings.events_representation,
                                            nr_events_window=settings.nr_events_window)
train_loader, _ = data_provider.get_train_loader(dataset=settings.dataset_name,
                                                 data_type="extract",
                                                 data_size=settings.batch_size,
                                                 num_workers=settings.num_workers,
                                                 train_transform=None,
                                                 repeat_augmentations=None,
                                                 events_representation=settings.events_representation,
                                                 drop_last=False,
                                                 nr_events_window=settings.nr_events_window)

feature_extractor.to(settings.device)


def load(feature_extractor, file_path):
    checkpoint = torch.load(file_path)
    feature_extractor.load_state_dict(checkpoint["backbone"])
    print("Checkpoint: {} is loaded".format(str(file_path)))


load(feature_extractor, settings.checkpoint_file)


def write_features(extracted, labels_, out_folder):
    for row in range(extracted.shape[0]):
        out_data = extracted[row, :]
        out_label = labels_[row].item()
        out_file = os.path.join(out_folder, str(out_label) + '/' + str(row))
        if not os.path.exists(os.path.dirname(out_file)):
            os.makedirs(os.path.dirname(out_file))
        np.save(out_file, out_data.detach().cpu().numpy())


def extract(feature_extractor, data_loader):
    feature_extractor.eval()
    extracted = None
    labels_ = None
    with torch.no_grad():
        for i_batch, data in enumerate(data_loader):
            samples, _, target = data
            if torch.cuda.is_available():
                samples, target = samples.cuda(), target.cuda()

            locations, features = Classifier.denseToSparse(samples)

            output = feature_extractor.forward([locations, features, samples.shape[0]]).detach()

            if extracted is not None and labels_ is not None:
                extracted = torch.cat((extracted, output), 0)
                labels_ = torch.cat((labels_, target), 0)
            else:
                extracted = output
                labels_ = target
        extracted = extracted.view(extracted.size(0), -1)
        labels_ = labels_.view(labels_.size(0), -1)
        assert extracted.shape[0] == labels_.shape[0]

        return extracted, labels_


def main():
    extracted, labels_ = extract(feature_extractor, train_loader)
    write_features(extracted, labels_, settings.out_folder + settings.dataset_name + "/training/")
    extracted, labels_ = extract(feature_extractor, test_loader)
    write_features(extracted, labels_, settings.out_folder + settings.dataset_name + "/testing/")


if __name__ == "__main__":
    main()
