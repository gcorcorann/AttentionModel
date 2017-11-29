#!/usr/bin/env python3
"""
Convolutional Neural Network Module.

@author:    Gary Corcoran
@date_create:   Nov. 28th, 2017

USAGE:  python cnn.py

"""
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CNN(nn.Module):
    """ CNN Modules. """
    def __init__(self):
        super(CNN, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        self.conv1 = nn.Conv2d(2, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 22 * 22, 2048)
        self.fc2 = nn.Linear(2048, 4)

    def forward(self, x):
        # max pooling over a (2,2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        # if he size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
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
            '../data/y_flow_small.npy')
    print('Inputs:', inputs.shape)
    print('Labels:', labels.shape)
    # create CNN object
    cnn = CNN()
    print(cnn)

    # define a loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(cnn.parameters(), lr=0.0001, momentum=0.9)

    # train the network
    print('Training Model...')
    for epoch in range(5):
        print('Epoch:', epoch)
        running_loss = 0.0
        # get the inputs
        for i, x_vid in enumerate(inputs):
            for j, x_frame in enumerate(x_vid):
                # create tensors
                x = np.swapaxes(x_frame, 0,2)
                x = torch.from_numpy(x)
                x = torch.unsqueeze(x, 0)
                y = labels[i]
                y = torch.LongTensor([int(y-1)])
                # wrap them in Variable
                input, label = Variable(x), Variable(y)
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                output = cnn(input)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.data[0]

        running_loss /= (i*j)
        print('\tLoss:', running_loss)
        running_loss = 0.0

    print('Finished Training')
                
if __name__ == '__main__':
    main()
