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
    X = np.load(X_path)
    y = np.load(y_path)
    return X, y

def random_training_example(X_train, y_train, batch_size):
    idx = np.random.permutation(X_train.shape[0])
    idx = idx[:batch_size]
    x = X_train[idx]
    y = y_train[idx]
    y -= 1  # [1-4] to [0-3]
    # need to use LongTensor
    y = np.asarray(y, dtype=np.int64)
    # reshape into nSamples x nChannels x Height x Width
    x = np.swapaxes(x, 1, -1)
    x_var = Variable(torch.from_numpy(x))
    y_var = Variable(torch.from_numpy(y))
    return x, x_var, y, y_var 

def train(cnn, X_train, y_train, criterion, optimizer):
    batch_size = 100
    print('Training Model...')
    for epoch in range(10):
        print('Epoch:', epoch)
        running_loss = 0.0
        # go through all training examples
        for i in range(X_train.shape[0]//batch_size):
            # get batch of random training examples
            # nSamples x nChannels x Height x Width
            x, x_var, y, y_var = random_training_example(X_train, y_train, 
                    batch_size)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            output = cnn(x_var)
            loss = criterion(output, y_var)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]

        print('Epoch Loss:', running_loss / X_train.shape[0] * batch_size)
        running_loss = 0

    print('Finished Training')

def main():
    """ Main Function. """
    print(__doc__)
    # read input data
    X_train, y_train = read_data('../data/X_flow_big.npy',
            '../data/y_flow_big.npy')
    # we only need to last frame for CNN
    X_train = X_train[:, -1].copy()
    print('X_train:', X_train.shape)
    print('y_train:', y_train.shape)

    # create CNN object
    cnn = CNN()
    print(cnn)
    # define a loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)

    # train the network
    train(cnn, X_train, y_train, criterion, optimizer)
                
if __name__ == '__main__':
    main()
