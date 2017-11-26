"""
RNN Training and Evaulation Module.

This module defines an RNN, trains it, and evaluates the performance.

@author: Gary Corcoran
@date_created: Nov. 20th, 2017

"""
import torch
import torch.nn as nn
from torch.autograd import Variable
from video_data import VideoData
from rnn_training import Training
import matplotlib.pyplot as plt


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


if __name__ == '__main__':
    # initialize video dataset parameters
    num_channels = 3
    width = height = 32
    vid_params = {'num_videos': 400, 'num_frames': 100, 'width': width, 
        'height': height, 'num_channels': num_channels, 'ratio': 0.75}
    # rnn parameters
    rnn_params = {'input_size': num_channels*width*height, 'hidden_size': 128,
        'output_size': 4}
    # training parameters
    learning_rate = 0.005
    num_epochs = 5
    # initialize video data object
    vids = VideoData('labels_gary.txt', **vid_params)
    # read and store videos
    X_train, y_train, X_test, y_test = vids.read_data(normalize=True, load=True)
    # initialize RNN object
    rnn = RNN(**rnn_params)
    # initialize training object
    tr = Training(rnn, X_train, y_train, learning_rate, num_epochs)
    training_losses = tr.train()
    # plot training losses
    plt.figure()
    plt.plot(training_losses)
    plt.show()
