#!/usr/bin/env python3
"""
Convolutional-Recurrent Neural Network Module.

@author:    Gary Corcoran
@date_created:  Dec. 2nd, 2017

USAGE:  python crnn.py

"""
import torch.nn as nn
from cnn_features import CNN_Features
from rnn import RNN

class CRNN(nn.Module):
    """ C-RNN Module. """
    def __init__(self):
        super(CRNN, self).__init__()
        self.cnn = CNN_Features()
        self.rnn = RNN(input_size=2048, hidden_size=256, output_size=4)

    def forward(self, x, hidden):
        x_feats = self.cnn.forward(x)
        output, hidden = self.rnn.forward(x_feats, hidden)
        return output, hidden

    def init_hidden(self, batch_size):
        """
        Initialize hidden units to zero.
        """
        return self.rnn.init_hidden(batch_size)

def test():
    """ Testing Function. """
    print(__doc__)
    # create CRNN object
    net = CRNN()
    print(net)

if __name__ == '__main__':
    test()
