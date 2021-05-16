"""
Most of the code in this file is taken
from https://github.com/mpatacchiola/self-supervised-relational-reasoning/blob/master/methods/simclr.py
Adaptations of the code are performed by Vadym Gryshchuk (vadym.gryshchuk@protonmail.com).

Backbone for a sparse convolutional neural network.
"""

import time
import collections

from torch.optim import Adam
from torch import nn
import torch
import tqdm

from utils import AverageMeter


class SimCLR(torch.nn.Module):
    def __init__(self, feature_extractor):
        super(SimCLR, self).__init__()

        self.net = nn.Sequential(collections.OrderedDict([
          ("feature_extractor", feature_extractor)
        ]))

        self.head = nn.Sequential(collections.OrderedDict([
          ("linear1",  nn.Linear(feature_extractor.feature_size, 256)),
          ("bn1",      nn.BatchNorm1d(256)),
          ("relu",     nn.LeakyReLU()),
          ("linear2",  nn.Linear(256, 64)),
        ]))

        self.optimizer = Adam([{"params": self.net.parameters(), "lr": 0.001},
                               {"params": self.head.parameters(), "lr": 0.001}])

        self.criterion = nn.CrossEntropyLoss()
        self.pbar = None

    @staticmethod
    def denseToSparse(dense_tensor):
        """
        Converts a dense tensor to a sparse vector.

        :param dense_tensor: BatchSize x SpatialDimension_1 x SpatialDimension_2 x ... x FeatureDimension
        :return locations: NumberOfActive x (SumSpatialDimensions + 1). The + 1 includes the batch index
        :return features: NumberOfActive x FeatureDimension
        """
        non_zero_indices = torch.nonzero(torch.abs(dense_tensor).sum(axis=-1))
        locations = torch.cat((non_zero_indices[:, 1:], non_zero_indices[:, 0, None]), dim=-1)

        select_indices = non_zero_indices.split(1, dim=1)
        features = torch.squeeze(dense_tensor[select_indices], dim=-2)

        return locations, features

    def return_loss_fn(self, x, t=0.5, eps=1e-8):
        # Taken from: https://github.com/pietz/simclr/blob/master/SimCLR.ipynb
        # Estimate cosine similarity
        n = torch.norm(x, p=2, dim=1, keepdim=True)
        x = (x @ x.t()) / (n * n.t()).clamp(min=eps)
        x = torch.exp(x / t)
        # Put positive pairs on the diagonal
        idx = torch.arange(x.size()[0])
        idx[::2] += 1
        idx[1::2] -= 1
        x = x[idx]
        # subtract the similarity of 1 from the numerator
        x = x.diag() / (x.sum(0) - torch.exp(torch.tensor(1 / t)))
        # NOTE: some implementation have used the loss `torch.mean(-torch.log(x))`,
        # but in preliminary experiments we saw that `-torch.log(x.mean())` is slightly
        # more effective (e.g. 77% vs 76% on CIFAR-10).
        return -torch.log(x.mean())

    def train(self, epoch, train_loader, nr_train_epochs):
        self.pbar = tqdm.tqdm(total=nr_train_epochs, unit='Batch', unit_scale=True,
                              desc='Epoch {}'.format(epoch))
        start_time = time.time()
        self.net.train()
        self.head.train()

        if epoch == 200:
            for i_g, g in enumerate(self.optimizer.param_groups):
                g["lr"] *= 0.1
                print("Group[" + str(i_g) + "] learning rate: " + str(g["lr"]))

        loss_meter = AverageMeter()
        statistics_dict = {}
        for i, (_, data_augmented, _) in enumerate(train_loader):
            data = torch.stack(data_augmented, dim=1)
            d = data.size()
            train_x = data.view(d[0]*2, d[2], d[3], d[4]).cuda()
            
            self.optimizer.zero_grad()
            if self.net.feature_extractor.name == "scnn":
                locations, features = self.denseToSparse(train_x)

                # forward pass in the backbone
                model_output = self.net([locations, features, train_x.shape[0]])
            else:
                model_output = self.net(train_x)
            tot_pairs = int(model_output.shape[0]*model_output.shape[0])
            embeddings = self.head(model_output)
            loss = self.return_loss_fn(embeddings)
            loss_meter.update(loss.item(), model_output.shape[0])
            loss.backward()
            self.optimizer.step()
            if i == 0:
                statistics_dict["batch_size"] = data.shape[0]
                statistics_dict["tot_pairs"] = tot_pairs
            self.pbar.set_postfix(TrainLoss=loss.data.cpu().numpy())
            self.pbar.update(1)
        self.pbar.close()
        elapsed_time = time.time() - start_time 
        print("Epoch [" + str(epoch) + "]"
               + "[" + str(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))) + "]"
               + " loss: " + str(loss_meter.avg)
               + "; batch-size: " + str(statistics_dict["batch_size"])
               + "; tot-pairs: " + str(statistics_dict["tot_pairs"]))
                             
        return loss_meter.avg, -loss_meter.avg

    def save(self, file_path="./checkpoint.dat"):
        feature_extractor_state_dict = self.net.feature_extractor.state_dict()
        head_state_dict = self.head.state_dict()
        optimizer_state_dict = self.optimizer.state_dict()
        torch.save({"backbone": feature_extractor_state_dict,
                    "head": head_state_dict,
                    "optimizer": optimizer_state_dict}, 
                    file_path)
        
    def load(self, file_path):
        checkpoint = torch.load(file_path)
        self.net.feature_extractor.load_state_dict(checkpoint["backbone"])
        self.head.load_state_dict(checkpoint["head"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
