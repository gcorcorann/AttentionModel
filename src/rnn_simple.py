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
    x_var = Variable(torch.from_numpy(x))
    x_var = x_var.view(99, 1, 100*100*2)
    y = int(y_train[r] - 1)
    y_var = Variable(torch.LongTensor([y]))
    return x, x_var, y, y_var

def train_instance(x_var, y_var, rnn, criterion, optimizer):
    """
    Train model on training instance.

    @param  x_var:  pytorch Variable containing single training instance
    @param  y_var:  pytorch Variable containing single training label
    """
    hidden = rnn.init_hidden()
    rnn.zero_grad()
    # for each frame in training example
    for frame in range(x_var.size()[0]):
        x_frame = x_var[frame]
        # pass through RNN
        output, hidden = rnn(x_frame, hidden)
    loss = criterion(output, y_var)
    loss.backward()
    optimizer.step()
    return output, loss.data[0]

def train(X_train, y_train, rnn):
    """
    Train Model.
    """
    print('Training Model...')
    # define a loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(rnn.parameters(), lr=0.0001, momentum=0.9)
    # keep track of losses
    current_loss = 0.0
    num_examples, num_frames = X_train.shape[:2]
    print('Num_examples:', num_examples)
    print('Num_frames:', num_frames)
    print()
    for epoch in range(5):
        print('Epoch:', epoch)
        for i in range(num_examples):
            # get random training example
            x, x_var, y, y_var = random_training_example(X_train, y_train)
            output, loss = train_instance(x_var, y_var, rnn, criterion,
                optimizer)
            current_loss += loss
        current_loss /= num_examples
        print('\tLoss:', current_loss)
        current_loss = 0
    print('Finished Training...')

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
    rnn = RNN(**rnn_params)
    print(rnn)
    print()
    
    # train model
    train(X_train, y_train, rnn)

if __name__ == '__main__':
    main()
