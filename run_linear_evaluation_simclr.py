"""
2021 Vadym Gryshchuk (vadym.gryshchuk@protonmail.com).

The main routine for training a single layer on top of SCNN.
"""

import argparse
import os
import random

import numpy as np
import torch
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
train_transform = data_provider.get_train_transforms("linear_evaluation", settings.dataset_name)
train_loader, _ = data_provider.get_train_loader(dataset=settings.dataset_name,
                                                 data_type="classifier",
                                                 data_size=settings.batch_size,
                                                 train_transform=train_transform,
                                                 repeat_augmentations=None,
                                                 num_workers=settings.num_workers,
                                                 drop_last=False,
                                                 events_representation=settings.events_representation,
                                                 nr_events_window=settings.nr_events_window)

test_loader = data_provider.get_test_loader(dataset=settings.dataset_name,
                                            data_size=settings.batch_size,
                                            events_representation=settings.events_representation,
                                            nr_events_window=settings.nr_events_window,
                                            num_workers=settings.num_workers)

if settings.backbone == "scnn":
    feature_extractor = SCNN(input_channels=10 if settings.events_representation == "voxel" else 2)
else:
    raise RuntimeError("[ERROR] the backbone " + str(settings.backbone) + " is not supported.")


def load_state_dict(feature_extractor, checkpoint):
    checkpoint_ = torch.load(checkpoint)
    state_dict = checkpoint_['backbone']
    from collections import OrderedDict
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        if 'module' in k:
            k = k[7:]  # remove `module.`
        new_state_dict[k] = v
    feature_extractor.load_state_dict(new_state_dict)
    print("Checkpoint: {} is loaded".format(str(checkpoint)))


def main():
    load_state_dict(feature_extractor, settings.checkpoint_file)
    model = Classifier(feature_extractor, num_classes)
    model.to(settings.device)
    if not os.path.exists("./checkpoint/" + str(settings.method_name) + "/" + str(settings.dataset_name)):
        os.makedirs("./checkpoint/" + str(settings.method_name) + "/" + str(settings.dataset_name))

    log_file = "./checkpoint/" + str(settings.method_name) + "/" + str(
        settings.dataset_name) + "/log_linear_evaluation" + settings.checkpoint_id + ".csv"
    mode = 'a' if os.path.exists(log_file) else 'w'
    if mode == 'w':
        with open(log_file, "w") as myfile:
            myfile.write("epoch,train_loss,train_score,val_loss,val_score_top1, val_score_top5" + "\n")

    for epoch in range(settings.epoch_start, settings.epochs):
        loss_train, accuracy_train = model.linear_evaluation(epoch, train_loader)
        checkpoint_path = "./checkpoint/" + str(settings.method_name) + "/" + str(
            settings.dataset_name) + "/" + settings.checkpoint_id + "_epoch_" + str(
            epoch + 1) + "_linear_evaluation.tar"
        if epoch % settings.checkpoint_save_period == 0 or epoch + 1 == settings.epochs:
            print("[INFO] Saving in:", checkpoint_path)
            model.save(checkpoint_path)
        loss_test, accuracy_test_top1, accuracy_test_top5 = model.test(test_loader)
        print("Test accuracy @1: " + str(accuracy_test_top1) + "%")
        print("Test accuracy @5: " + str(accuracy_test_top5) + "%")
        with open(log_file, "a") as myfile:
            myfile.write(str(epoch) + "," + str(loss_train) + "," + str(accuracy_train) + "," + str(loss_test) + ","
                         + str(accuracy_test_top1) + "," + str(accuracy_test_top5) + "\n")


if __name__ == "__main__":
    main()
