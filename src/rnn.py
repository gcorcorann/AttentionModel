#!/usr/bin/env python3
"""
Defines RNN Model.

@author: Gary Corcoran
@date_created: Nov. 28th, 2017

USAGE:  python rnn.py

Keys:
"""
import torch
import torch.nn as nn
from torch.autograd import Variable

class RNN(nn.Module):
    """ RNN based on image captioning. """
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize RNN class.

        @param  input_size:     input dimension size
        @param  hidden_size:    number of hidden nodes
        @param  output_size:    number of output units
        """
        super(RNN, self).__init__()
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

def main():
    """ Main Function. """
    # rnn parameters
    rnn_params = {'input_size': width*height*2, 'hidden_size': 256,
        'output_size': 4}
    # initialize RNN object
    rnn = RNN(**rnn_params)
    print(rnn)
    print()

if __name__ == '__main__':
    main()
