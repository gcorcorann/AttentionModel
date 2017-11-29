#!/usr/bin/env python3
"""
RNN Training and Evaulation Module.

This module defines an RNN, trains it, and evaluates the performance.

@author: Gary Corcoran
@date_created: Nov. 20th, 2017

USAGE:  python rnn.py

Keys:
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
from cnn_features import CNN_Features

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

def sequence_to_tensor(x):
    """
    Converts data sequence to tensor.

    @param  x:  array consisting of video data [num_frames, num_feats]

    @return x:  tensor of dimensions [num_frames, 1, num_feats]
    """
    # add dimension for batch size placement
    x = np.expand_dims(x, axis=1)
    return torch.from_numpy(x)

def random_training_example(X_train, y_train):
    """
    Retrieve random training example.

    @param  X_train:    array of training inputs
    @param  y_train:    array of training labels

    @return x:      array containing random training example
    @return x_var:  pytorch Variable containing random training example
    @return y:      integer containing random training label
    @return y_var:  pytorch Variable containing random training label
    """
    r = random.randint(0, X_train.shape[0]-1)
    x = X_train[r]
    x_var = Variable(sequence_to_tensor(x))
    y = int(y_train[r] - 1)
    y_var = Variable(torch.LongTensor(y))
    return x, x_var, y, y_var

def train(X_train, y_train, rnn, cnn):
    """
    Train Model.
    """
    print('Training Model...')
    # keep track of losses
    current_loss = 0.0
    num_examples, num_frames = X_train.shape[:2]
    print('Num_examples:', num_examples)
    print('Num_frames:', num_frames)
    print()
    for epoch in range(2):
        print('Epoch:', epoch)
        for i in range(num_examples):
            # get random training example
            x, x_var, y, y_var = random_training_example(X_train, y_train)
            print(x.shape, y)
                


def main():
    """ Main Function. """
    print(__doc__)
    # load small flow dataset
    X_train, y_train = read_data('../data/X_flow_small.npy',
            '../data/y_small.npy')
    print('X_train:', X_train.shape)
    print('y_train:', y_train.shape)
    print()
    # create CNN object
    cnn = CNN_Features()
    print(cnn)
    params = list(cnn.parameters())
    # last layer in CNN module
    rnn_input_size = params[-1].size()[0]
    # rnn parameters
    rnn_params = {'input_size': rnn_input_size, 'hidden_size': 512,
        'output_size': 4}
    # initialize RNN object
    rnn = RNN(**rnn_params)
    print(rnn)
    print()
    
    # train model
    train(X_train, y_train, rnn, cnn)

if __name__ == '__main__':
    main()
