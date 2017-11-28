#!/usr/bin/env python3
"""
RNN Training and Evaulation Module.

This module defines an RNN, trains it, and evaluates the performance.

@author: Gary Corcoran
@date_created: Nov. 20th, 2017

USAGE:  python rnn.py

Keys:
"""
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

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

    def init_hidden(self):
        """
        Initialize hidden units to zero.
        """
        return Variable(torch.zeros(1, self.hidden_size))


# HELPER FUNCTIONS #
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
    # rnn parameters
    width = height = 100
    num_channels = 2
    rnn_params = {'input_size': width*height*num_channels, 'hidden_size': 128,
        'output_size': 4}
    # training parameters
    learning_rate = 0.005
    num_epochs = 5
    # initialize RNN object
    rnn = RNN(**rnn_params)
    print(rnn)
    # read input data
    X, y = read_data('../data/X_flow.npy', '../data/y.npy')
    print(X.shape)
    print(y.shape)

if __name__ == '__main__':
    main()
