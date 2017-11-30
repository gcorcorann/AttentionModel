#!/usr/bin/env python3
"""
RNN Training and Evaulation Module.

This module defines an RNN, trains it, and evaluates the performance.

@author: Gary Corcoran
@date_created: Nov. 28th, 2017

USAGE:  python rnn_simple.py

Keys:
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

class RNN(nn.Module):
    """ Vanilla RNN. """
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize RNN class.

        @param  input_size:     input dimension size
        @param  hidden_size:    number of hidden nodes
        @param  output_size:    number of output units
        """
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        """
        Forward pass through network.
        """
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.h2o(hidden)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self, batch_size):
        """
        Initialize hidden units to zero.
        """
        return Variable(torch.zeros(batch_size, self.hidden_size))

class RNN2(nn.Module):
    """ RNN based on image captioning. """
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize RNN class.

        @param  input_size:     input dimension size
        @param  hidden_size:    number of hidden nodes
        @param  output_size:    number of output units
        """
        super(RNN2, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        """
        Forward pass through network.
        """
        input_combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self, batch_size):
        """
        Initialize hidden units to zero.
        """
        return Variable(torch.zeros(batch_size, self.hidden_size))


# HELPER FUNCTIONS #
import random
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
    """
    Retrieve random training example.

    @param  X_train:    array of training inputs
    @param  y_train:    array of training labels

    @return x:      array containing random training example
    @return x_var:  pytorch Variable containing random training example
    @return y:      integer containing random training label
    @return y_var:  pytorch Variable containing random training label
    """
    idx = np.random.permutation(X_train.shape[0])
    idx = idx[:batch_size]
    x = X_train[idx]
    y = y_train[idx]
    y -= 1  # [1-4] to [0-3]
    # need to use LongTensor
    y = np.asarray(y, dtype=np.int64)
    # reshape into nSequence x nBatch x nFeatures
    x = np.swapaxes(x, 1, 0)
    x = np.reshape(x, (99, batch_size, -1))
    x_var = Variable(torch.from_numpy(x))
    y_var = Variable(torch.from_numpy(y))
    return x, x_var, y, y_var

def train(rnn, X_train, y_train, criterion, optimizer):
    batch_size = 5
    print('Training Model...')
    for epoch in range(3):
        print('Epoch:', epoch)
        running_loss = 0.0
        # go through all training examples
        for i in range(X_train.shape[0]//batch_size):
            print(i)
            # get batch of random training examples
            # nSequence x nBatch x nFeatures
            x, x_var, y, y_var = random_training_example(X_train, y_train, 
                    batch_size)
            # zero the parameter gradients
            optimizer.zero_grad()
            # zero initial hidden state
            hidden = rnn.init_hidden(batch_size)
            # for each frame in sequence
            for j in range(x_var.size()[0]):
                # pass through RNN
                output, hidden = rnn(x_var[j], hidden)
            loss = criterion(output, y_var)
            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]
        
        # print statistics
        print('Epoch Loss:', running_loss / X_train.shape[0] * batch_size)
        running_loss = 0

    print('Finished Training.')

def train2(rnn, X_train, y_train, criterion, optimizer):
    batch_size = 5
    print('Training Model...')
    for epoch in range(5):
        print('Epoch:', epoch)
        running_loss = 0.0
        # go through all training examples
        for i in range(X_train.shape[0]//batch_size):
            # get batch of random training examples
            # nSequence x nBatch x nFeatures
            x, x_var, y, y_var = random_training_example(X_train, y_train, 
                    batch_size)
            # zero the parameter gradients
            optimizer.zero_grad()
            # zero initial hidden state
            hidden = rnn.init_hidden(batch_size)
            # for each frame in sequence
            loss = 0
            for j in range(x_var.size()[0]):
                # pass through RNN
                output, hidden = rnn(x_var[j], hidden)
                loss += criterion(output, y_var)
            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]
        
        # print statistics
        print('Epoch Loss:', running_loss / X_train.shape[0] * batch_size)
        running_loss = 0

    print('Finished Training.')

def main():
    """ Main Function. """
    print(__doc__)
    # load small flow dataset
    X_train, y_train = read_data('../data/X_flow_small.npy',
            '../data/y_flow_small.npy')
    print('X_train:', X_train.shape)
    print('y_train:', y_train.shape)
    print()
    # rnn parameters
    rnn_params = {'input_size': 100*100*2, 'hidden_size': 256,
        'output_size': 4}
    # initialize RNN object
    rnn = RNN2(**rnn_params)
    print(rnn)
    print()

    # define a loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(rnn.parameters(), lr=0.000001, momentum=0.9)
    # train model
    train2(rnn, X_train, y_train, criterion, optimizer)

if __name__ == '__main__':
    main()
