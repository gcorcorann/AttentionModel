#!/usr/bin/env python3
"""
RNN Training Module.

This module trains an cnn+rnn model.

@author: Gary Corcoran
@date_created: Nov. 20th, 2017

USAGE:  python rnn.py

Keys:
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from cnn_features import CNN_Features
from rnn import RNN
import random
import numpy as np

# HELPER FUNCTIONS #
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
    y = np.asarray(y, dtype=np.int64)
    # reshape into nSamples x nChannels x Height x Width
    x = np.swapaxes(x, 0, 1)
    x = np.moveaxis(x, -1, 2)
    x_var = Variable(torch.from_numpy(x))
    y_var = Variable(torch.from_numpy(y))
    return x, x_var, y, y_var

def train(cnn, rnn, X_train, y_train, criterion, optimizer):
    batch_size = 2
    print('Training Model...')
    for epoch in range(5):
        print('Epoch:', epoch)
        running_loss = 0.0
        # go through all training examples
        for i in range(X_train.shape[0]//batch_size):
            # get random training examples
            x, x_var, y, y_var = random_training_example(X_train, y_train,
                    batch_size)
            # zero the parameter gradients
            optimizer.zero_grad()
            # zero initial hidden state
            hidden = rnn.init_hidden(batch_size)
            # for each frame in sequence
            for j in range(x_var.size()[0]):
                # pass through CNN
                cnn_feats = cnn(x_var[j])
                # pass through RNN
                output, hidden = rnn(cnn_feats, hidden)
            # back + optimize
            loss = criterion(output, y_var)
            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]

        # print statistics
        print('Epoch loss:', running_loss / X_train.shape[0] * batch_size)

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
    # create CNN object
    cnn = CNN_Features()
    print(cnn)
    params = list(cnn.parameters())
    # last layer in CNN module
    rnn_input_size = params[-1].size()[0]
    # rnn parameters
    rnn_params = {'input_size': rnn_input_size, 'hidden_size': 256,
        'output_size': 4}
    # initialize RNN object
    rnn = RNN(**rnn_params)
    print(rnn)
    print()

    # define a loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.SGD([
        {'params': cnn.parameters(), 'lr': 1e-3, 'momentum': 0.9},
        {'params': rnn.parameters(), 'lr': 1e-3, 'momentum': 0.9}
        ])
    
    # train model
    train(cnn, rnn, X_train, y_train, criterion, optimizer)

if __name__ == '__main__':
    main()
