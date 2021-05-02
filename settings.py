"""
2021 Vadym Gryshchuk (vadym.gryshchuk@protonmail.com).

Parameters.

"""

import os
import yaml
import torch


class Settings:
    def __init__(self, settings_yaml):
        assert os.path.isfile(settings_yaml), settings_yaml

        with open(settings_yaml, 'r') as stream:
            settings = yaml.load(stream, yaml.Loader)

            # --- hardware ---
            hardware = settings['hardware']
            gpu_device = hardware['gpu_device']
            self.visible_gpu = gpu_device
            self.device = torch.device("cpu") if gpu_device == "cpu" else torch.device("cuda:" + str(gpu_device))
            self.num_workers = hardware['num_workers']
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.visible_gpu)

            # --- dataset ---
            dataset = settings['dataset']
            self.dataset_name = dataset['name']
            self.events_representation = dataset['event_representation']
            self.nr_events_window = dataset['nr_events_window']

            assert self.events_representation in ["voxel", "histogram"]

            # --- method ---
            method = settings['method']
            self.method_name = method['name']
            self.backbone = method['backbone']
            self.seed = method['seed']
            self.checkpoint_save_period = method['checkpoint_save_period']
            self.epochs = method['epochs']
            self.batch_size = method['batch_size']

            if self.backbone == "plstm":
                try:
                    assert self.method_name == "classifier"
                    assert self.nr_events_window == 0
                except Exception as error:
                    print("Phased LSTM can be used only with the classifier method. Window must be set to zero", error)
                    raise

            # --- checkpoint ---
            checkpoint = settings['checkpoint']
            self.epoch_start = checkpoint['epoch_start']
            self.checkpoint_file = checkpoint['file']

            self.checkpoint_id = str(self.method_name) + "_" + str(checkpoint['id']) + "_" + str(
                self.dataset_name) + "_" + str(self.backbone) + "_seed_" + str(
                self.seed) + "_batch_" + str(self.batch_size)

            # --- feature extraction ---
            out_folder = settings['feature_extraction']
            self.out_folder = out_folder['out_folder']

