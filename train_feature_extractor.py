"""
2021 Vadym Gryshchuk (vadym.gryshchuk@protonmail.com).

The main routine for training SCNN or Phased LSTM.
"""

import argparse
import os
import random
import sys

import numpy as np
import torch.optim
from torch.backends import cudnn
from torch.cuda import manual_seed, manual_seed_all

from backbones.phased_lstm import PhasedLSTM
from backbones.scnn import SCNN, SCNNParallel
from data_provider import DataProvider
from method.classifier import Classifier
from method.simclr import SimCLR
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

if settings.backbone == "scnn":
    feature_extractor = SCNN(input_channels=10 if settings.events_representation == "voxel" else 2)
    feature_extractor = SCNNParallel(feature_extractor).to(settings.device)
    print(feature_extractor)
elif settings.backbone == "plstm":
    feature_extractor = PhasedLSTM(3, 128, 0.001, 0.05, 1, 1000000).to(settings.device)
    print(feature_extractor)
else:
    raise RuntimeError("The backbone {} is not recognized".format(str(settings.backbone)))

total_parameterss = sum(param.numel() for param in feature_extractor.parameters() if param.requires_grad)
print("Total trainable parameters: " + str(total_parameterss))

data_provider = DataProvider(settings.seed)
num_classes = data_provider.get_num_classes(settings.dataset_name)

if settings.method_name == "classifier":

    model = Classifier(feature_extractor, num_classes, tot_epochs=settings.epochs)

    if settings.backbone == "plstm":
        data_type = "raw"
        train_transform = None
        event_mode = None
        from data_provider import collate_events

        collate_fn = collate_events
    else:
        data_type = "classifier"
        event_mode = settings.events_representation
        collate_fn = None
        train_transform = data_provider.get_train_transforms("classifier", settings.dataset_name)

    test_loader = data_provider.get_test_loader(dataset=settings.dataset_name,
                                                data_size=settings.batch_size,
                                                num_workers=settings.num_workers,
                                                events_representation=event_mode,
                                                collate_fn=collate_fn,
                                                nr_events_window=settings.nr_events_window)
    train_loader, _ = data_provider.get_train_loader(dataset=settings.dataset_name,
                                                     data_type=data_type,
                                                     data_size=settings.batch_size,
                                                     train_transform=train_transform,
                                                     repeat_augmentations=None,
                                                     num_workers=settings.num_workers,
                                                     drop_last=False,
                                                     events_representation=event_mode,
                                                     nr_events_window=settings.nr_events_window)
elif settings.method_name == "simclr":
    model = SimCLR(feature_extractor)
    train_transform = data_provider.get_train_transforms(settings.method_name, settings.dataset_name)
    train_loader, _ = data_provider.get_train_loader(dataset=settings.dataset_name,
                                                     data_type="simclr",
                                                     data_size=settings.batch_size,
                                                     train_transform=train_transform,
                                                     repeat_augmentations=2,
                                                     num_workers=settings.num_workers,
                                                     drop_last=False,
                                                     events_representation=settings.events_representation,
                                                     nr_events_window=settings.nr_events_window)
else:
    raise RuntimeError("The method {} is not recognized.".format(str(settings.method_name)))

model.to(settings.device)
if settings.checkpoint_file != "":
    model.load(settings.checkpoint_file)
    print("Checkpoint: {} is loaded".format(str(settings.checkpoint_file)))


def main():
    # Create checkpoint path:
    if not os.path.exists("./checkpoint/" + str(settings.method_name) + "/" + str(settings.dataset_name)):
        os.makedirs("./checkpoint/" + str(settings.method_name) + "/" + str(settings.dataset_name))

    # Create log file:
    log_file = "./checkpoint/" + str(settings.method_name) + "/" + str(
        settings.dataset_name) + "/log_" + settings.checkpoint_id + ".csv"
    mode = 'a' if os.path.exists(log_file) else 'w'
    if mode == 'w':
        with open(log_file, mode) as myfile:
            myfile.write("epoch,loss,score" + "\n")

    # Create args file:
    commandline_settings = "./checkpoint/" + str(settings.method_name) + "/" + str(
        settings.dataset_name) + "/args_" + settings.checkpoint_id + ".csv"
    with open(commandline_settings, 'w') as f:
        f.write('\n'.join(sys.argv[1:]))
    for epoch in range(settings.epoch_start, settings.epochs):
        loss_train, accuracy_train = model.train(epoch, train_loader, data_provider.nr_train_epochs)
        with open(log_file, "a") as myfile:
            myfile.write(str(epoch) + "," + str(loss_train) + "," + str(accuracy_train) + "\n")
        checkpoint_path = "./checkpoint/" + str(settings.method_name) + "/" + str(
            settings.dataset_name) + "/" + settings.checkpoint_id + "_epoch_" + str(epoch + 1) + ".tar"
        if epoch % settings.checkpoint_save_period == 0 or epoch + 1 == settings.epochs:
            print("Saving in:", checkpoint_path)
            model.save(checkpoint_path)
    if settings.method_name == "classifier":
        loss_test, accuracy_test_top1, accuracy_test_top5 = model.test(test_loader)
        print("Test loss: " + str(loss_test))
        print("Test accuracy @1: " + str(accuracy_test_top1) + "%")
        print("Test accuracy @5: " + str(accuracy_test_top5) + "%")


if __name__ == "__main__":
    main()
