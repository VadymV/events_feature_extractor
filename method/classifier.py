"""
Most of the code in this file is taken
from https://github.com/mpatacchiola/self-supervised-relational-reasoning/blob/master/methods/standard.py
Adaptations of the code are performed by Vadym Gryshchuk (vadym.gryshchuk@protonmail.com).

Backbone for a sparse convolutional neural network.
"""

import time

import numpy as np
import torch
import tqdm
from torch import nn
from torch.optim import SGD, Adam

import utils
from utils import AverageMeter


class Classifier(torch.nn.Module):
    def __init__(self, feature_extractor, num_classes, tot_epochs=200):
        super(Classifier, self).__init__()
        self.num_classes = num_classes
        self.tot_epochs = tot_epochs
        self.feature_extractor = feature_extractor
        feature_size = feature_extractor.feature_size
        self.classifier = nn.Linear(feature_size, num_classes)
        self.ce = torch.nn.CrossEntropyLoss()
        if feature_extractor.name != "plstm":
            self.optimizer = SGD([{"params": self.feature_extractor.parameters(), "lr": 0.01, "momentum": 0.9},
                                  {"params": self.classifier.parameters(), "lr": 0.01, "momentum": 0.9}])
        else:
            self.optimizer = Adam([{"params": self.feature_extractor.parameters(), "lr": 0.003},
                                   {"params": self.classifier.parameters(), "lr": 0.003}])
        self.optimizer_lineval = Adam([{"params": self.classifier.parameters(), "lr": 0.001}])
        self.optimizer_finetune = Adam(
            [{"params": self.feature_extractor.parameters(), "lr": 0.001, "weight_decay": 1e-5},
             {"params": self.classifier.parameters(), "lr": 0.0001, "weight_decay": 1e-5}])
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

    def forward(self, x, detach=False):
        if detach:
            out = self.feature_extractor(x).detach()
        else:
            out = self.feature_extractor(x)
        out = self.classifier(out)
        return out

    def train(self, epoch, train_loader, nr_train_epochs):
        self.pbar = tqdm.tqdm(total=nr_train_epochs, unit='Batch', unit_scale=True,
                              desc='Epoch {}'.format(epoch))
        start_time = time.time()
        self.feature_extractor.train()
        self.classifier.train()
        if epoch == int(self.tot_epochs * 0.5) or epoch == int(self.tot_epochs * 0.75):
            for i_g, g in enumerate(self.optimizer.param_groups):
                g["lr"] *= 0.1  # divide by 10
                print("Group[" + str(i_g) + "] learning rate: " + str(g["lr"]))
        loss_meter = AverageMeter()
        accuracy_meter = AverageMeter()
        for i, (data, additional_info, target) in enumerate(train_loader):
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            self.optimizer.zero_grad()

            if self.feature_extractor.name == "scnn":
                locations, features = self.denseToSparse(data)

                output = self.forward([locations, features, data.shape[0]])
            elif self.feature_extractor.name == "plstm":
                additional_info = additional_info.cuda()
                output = self.forward((data, additional_info))
            else:
                output = self.forward(data)

            loss = self.ce(output, target)
            loss_meter.update(loss.item(), len(target))
            loss.backward()
            self.optimizer.step()
            pred = output.argmax(-1)
            correct = pred.eq(target.view_as(pred)).cpu().sum()
            accuracy = (100.0 * correct / float(len(target)))
            accuracy_meter.update(accuracy.item(), len(target))
            self.pbar.set_postfix(TrainLoss=loss.data.cpu().numpy(), TrainAcc=accuracy.data.cpu().numpy())
            self.pbar.update(1)
        self.pbar.close()
        elapsed_time = time.time() - start_time
        print("Epoch [" + str(epoch) + "]"
              + "[" + str(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))) + "]"
              + " loss: " + str(loss_meter.avg)
              + "; acc: " + str(accuracy_meter.avg) + "%")
        return loss_meter.avg, accuracy_meter.avg

    def linear_evaluation(self, epoch, train_loader):
        self.feature_extractor.eval()
        self.classifier.train()
        minibatch_iter = tqdm.tqdm(train_loader, desc=f"(Epoch {epoch}) Minibatch")
        if epoch == int(self.tot_epochs * 0.5) or epoch == int(self.tot_epochs * 0.75):
            for i_g, g in enumerate(self.optimizer.param_groups):
                g["lr"] *= 0.1  # divide by 10
                print("Group[" + str(i_g) + "] learning rate: " + str(g["lr"]))
        loss_meter = AverageMeter()
        accuracy_meter = AverageMeter()
        for data, _, target in minibatch_iter:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            self.optimizer_lineval.zero_grad()
            if self.feature_extractor.name == "scnn":
                locations, features = self.denseToSparse(data)

                output = self.forward([locations, features, data.shape[0]], detach=True)
            else:
                output = self.forward(data, detach=True)
            loss = self.ce(output, target)
            loss_meter.update(loss.item(), len(target))
            loss.backward()
            self.optimizer_lineval.step()
            pred = output.argmax(-1)
            correct = pred.eq(target.view_as(pred)).cpu().sum()
            accuracy = (100.0 * correct / float(len(target)))
            accuracy_meter.update(accuracy.item(), len(target))
            minibatch_iter.set_postfix({"loss": loss_meter.avg, "acc": accuracy_meter.avg})
        return loss_meter.avg, accuracy_meter.avg

    def test(self, test_loader):
        self.feature_extractor.eval()
        self.classifier.eval()
        loss_meter = AverageMeter()
        accuracy_top1_meter = AverageMeter()
        accuracy_top5_meter = AverageMeter()
        y_true = None
        y_pred = None
        with torch.no_grad():
            for data, additional_info, target in test_loader:
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()

                if self.feature_extractor.name == "scnn":
                    locations, features = self.denseToSparse(data)

                    output = self.forward([locations, features, data.shape[0]], detach=True)
                elif self.feature_extractor.name == "plstm":
                    additional_info = additional_info.cuda()
                    output = self.forward((data, additional_info), detach=True)
                else:
                    output = self.forward(data, detach=True)
                loss = self.ce(output, target)
                acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
                loss_meter.update(loss.item(), len(target))
                accuracy_top1_meter.update(acc1.item(), len(target))
                accuracy_top5_meter.update(acc5.item(), len(target))
                if y_true is None and y_pred is None:
                    y_true = target.cpu().numpy()
                    y_pred = output.argmax(-1).cpu().numpy()
                else:
                    y_true = np.hstack((y_true, target.cpu().numpy()))
                    y_pred = np.hstack((y_pred, output.argmax(-1).cpu().numpy()))
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        cm = confusion_matrix(y_true, y_pred)
        g = sns.heatmap(cm / np.sum(cm, axis=1), cmap='YlGnBu')
        g.figure.savefig("./images/cm_classes-{}.pdf".format(len(np.unique(y_true))),
                         format="pdf",
                         bbox_inches="tight")
        return loss_meter.avg, accuracy_top1_meter.avg, accuracy_top5_meter.avg

    def return_embeddings(self, data_loader, portion=0.5):
        self.feature_extractor.eval()
        embeddings_list = []
        target_list = []
        with torch.no_grad():
            for i, (data, target) in enumerate(data_loader):
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()
                features = self.feature_extractor(data)
                embeddings_list.append(features)
                target_list.append(target)
                if i >= int(len(data_loader) * portion):
                    break
        return torch.cat(embeddings_list, dim=0).cpu().detach().numpy(), torch.cat(target_list,
                                                                                   dim=0).cpu().detach().numpy()

    def save(self, file_path="./checkpoint.dat"):
        state_dict = self.classifier.state_dict()
        feature_extractor_state_dict = self.feature_extractor.state_dict()
        optimizer_state_dict = self.optimizer.state_dict()
        optimizer_lineval_state_dict = self.optimizer_lineval.state_dict()
        optimizer_finetune_state_dict = self.optimizer_finetune.state_dict()
        torch.save({"classifier": state_dict,
                    "backbone": feature_extractor_state_dict,
                    "optimizer": optimizer_state_dict,
                    "optimizer_lineval": optimizer_lineval_state_dict,
                    "optimizer_finetune": optimizer_finetune_state_dict},
                   file_path)

    def load(self, file_path):
        checkpoint = torch.load(file_path)
        self.classifier.load_state_dict(checkpoint["classifier"])
        self.feature_extractor.load_state_dict(checkpoint["backbone"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.optimizer_lineval.load_state_dict(checkpoint["optimizer_lineval"])
        self.optimizer_finetune.load_state_dict(checkpoint["optimizer_finetune"])
