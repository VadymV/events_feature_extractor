"""
Most of the code in this file is taken
from https://github.com/uzh-rpg/rpg_asynet/blob/master/models/facebook_sparse_vgg.py.
Adaptations of the code are performed by Vadym Gryshchuk (vadym.gryshchuk@protonmail.com).

Backbone for a sparse convolutional neural network.
"""

import sparseconvnet as scn
import torch
import torch.nn as nn


class SCNNParallel(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class SCNN(nn.Module):
    def __init__(self, input_channels=10):
        super(SCNN, self).__init__()
        self.feature_size = 1536
        self.name = "scnn"

        sparse_out_channels = 256
        self.sparseModel = scn.SparseVggNet(2, nInputPlanes=input_channels, layers=[
            ['C', 16], ['C', 16], 'MP',
            ['C', 32], ['C', 32], 'MP',
            ['C', 64], ['C', 64], 'MP',
            ['C', 128], ['C', 128], 'MP',
            ['C', 256], ['C', 256], 'MP',
            ['C', 512]]

                                            ).add(
            scn.Convolution(2, 512, sparse_out_channels, 3, filter_stride=2, bias=False)
            ).add(scn.BatchNormReLU(sparse_out_channels)
                  ).add(scn.SparseToDense(2, sparse_out_channels))

        cnn_spatial_output_size = [2, 3]
        self.spatial_size = self.sparseModel.input_spatial_size(torch.LongTensor(cnn_spatial_output_size))
        self.inputLayer = scn.InputLayer(dimension=2, spatial_size=self.spatial_size, mode=2)
        self.linear_input_features = cnn_spatial_output_size[0] * cnn_spatial_output_size[1] * sparse_out_channels

    def forward(self, x):
        x = self.inputLayer(x)
        x = self.sparseModel(x)
        x = x.view(-1, self.linear_input_features)

        return x
