"""
2021 Vadym Gryshchuk (vadym.gryshchuk@protonmail.com).

The main routine for testing SCNN or Phased LSTM.
"""

import argparse
import random

import numpy as np
import torch.optim
from torch.backends import cudnn
from torch.cuda import manual_seed, manual_seed_all

from backbones.scnn import SCNN
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

data_provider = DataProvider(settings.seed)
num_classes = data_provider.get_num_classes(settings.dataset_name)

if settings.backbone == "scnn":
    feature_extractor = SCNN(input_channels=10 if settings.events_representation == "voxel" else 2)
    print(feature_extractor)
else:
    raise RuntimeError("The backbone {} is not recognized".format(str(settings.backbone)))

test_loader = data_provider.get_test_loader(dataset=settings.dataset_name,
                                            data_size=settings.batch_size,
                                            events_representation=settings.events_representation,
                                            nr_events_window=settings.nr_events_window,
                                            num_workers=settings.num_workers)


def main():
    model = Classifier(feature_extractor, num_classes)
    model.load(settings.checkpoint_file)
    print("Checkpoint: {} is loaded".format(str(settings.checkpoint_file)))

    model.to(settings.device)

    loss_test, accuracy_test_top1, accuracy_test_top5 = model.test(test_loader)
    print("Test loss:", str(loss_test) + "%")
    print("Test accuracy @1:", str(accuracy_test_top1) + "%")
    print("Test accuracy @5:", str(accuracy_test_top5) + "%")
    print("===========================")


if __name__ == "__main__":
    main()
