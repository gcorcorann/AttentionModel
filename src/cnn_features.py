#!/usr/bin/env python3
"""
Convolutional Neural Network Module.

@author:    Gary Corcoran
@date_create:   Nov. 28th, 2017

USAGE:  python cnn_features.py

"""
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CNN_Features(nn.Module):
    """ CNN Features Modules. """
    def __init__(self):
        super(CNN_Features, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        self.conv1 = nn.Conv2d(2, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 22 * 22, 2048)

    def forward(self, x):
        # max pooling over a (2,2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        # if he size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        return x

    def num_flat_features(self, x):
        # all dimensions expect the batch dimension
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# HELPER FUNCTION #
import numpy as np

def read_data(X_path, y_path):
    """
    Read and return data.

    @param  X_path:     path to input data
    @param  y_path:     path to input labels

    @return X:          array of input data
    @return y:          array of input labels
    """
    print(X_path, y_path)
    X = np.load(X_path)
    y = np.load(y_path)
    return X, y

def main():
    """ Main Function. """
    print(__doc__)
    # read input data
    inputs, labels = read_data('../data/X_flow_small.npy',
            '../data/y_small.npy')

    print('Inputs:', inputs.shape)
    print('Labels:', labels.shape)
    # create CNN object
    net = CNN_Features()
    print(net)

if __name__ == '__main__':
    main()
