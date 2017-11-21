"""
RNN Training and Evaulation Module.

This module defines an RNN, trains it, and evaluates the performance.

@author: Gary Corcoran
@date: Nov. 20th, 2017

"""
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from video_data import VideoData
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
        self.n_iters = 10000
        self.learning_rate = 0.005
        self.criterion = nn.NLLLoss()

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

    def _sequence_to_tensor(self, x):
        """
        Converts data sequence to tensor.

        @param  x:  numpy array consisting of video data
                    [num_frames, num_feats]

        @return x:  tensor of dimension [num_frames, 1, num_feats]
        """
        # add dimension for batch size placeholder
        x = np.expand_dims(x, axis=1)
        tensor = torch.from_numpy(x)
        return tensor

    def __random_training_example(self, X_train, y_train):
        """
        Get random training example.

        @return x:  pytorch Variable consisting of one training instance
        @return y:  pytorch Variable consisting of one training label
        """
        r = random.randint(0, X_train.shape[0]-1)
        # random training example
        x = Variable(self._sequence_to_tensor(X_train[r]))
        y = Variable(torch.LongTensor([int(y_train[r] - 1)]))
        return x, y

    def __train_instance(self, x_tensor, y_tensor):
        # initialize hidden units to zero
        hidden = rnn.init_hidden()
        # zero gradients
        self.zero_grad()
        # for each frame in training instance 
        for i in range(x_tensor.size()[0]):
            output, hidden = rnn(x_tensor[i], hidden)
        # compute loss
        loss = self.criterion(output, y_tensor)
        # compute backprop
        loss.backward()
        # add parameters' gradients to their values, multiplied by learning rate
        for p in self.parameters():
            p.data.add_(-self.learning_rate, p.grad.data)
        return output, loss.data[0]

    def label_from_output(self, output):
        top_n, top_i = output.data.topk(1)
        label = top_i[0][0] + 1
        return label

    def train(self, X_train, y_train, n_iters, learning_rate):
        print('Training...')
        self.n_iters = n_iters
        self.learning_rate = learning_rate
        # keep track of losses
        current_loss = 0
        all_losses = []
        print_every = 100
        plot_every = 50 
        X_train = np.float32(X_train)
        for iter in range(1, n_iters + 1):
            print(iter)
            # get random training example
            x_tensor, y_tensor = self.__random_training_example(X_train,
                    y_train)
            output, loss = self.__train_instance(x_tensor, y_tensor)
            current_loss += loss
            # print iter number, loss, name, and guess
            if iter % plot_every == 0:
                all_losses.append(current_loss / plot_every)
                current_loss = 0
        return all_losses

if __name__ == '__main__':
    # data parameters
    num_videos = 400
    num_frames = 100
    width = height = 32
    num_channels = 3
    # rnn parameters
    n_categories = 4
    n_hidden = 128
    n_iters = 1000
    learning_rate = 0.005
    # initialize video data object
    vids = VideoData('labels_gary.txt', num_videos, num_frames, width, height,
            num_channels)
    # read and store videos
    vids.read_data()
    vids.normalize_data()
    X_train, y_train = vids.X, vids.y
    # initialize RNN object
    rnn = RNN(num_channels*width*height, n_hidden, n_categories)
    all_losses = rnn.train(X_train, y_train, n_iters, learning_rate)
    print(all_losses)

    plt.figure()
    plt.plot(all_losses)
    plt.show()

#   input = Variable(rnn.sequence_to_tensor(X_train[0], num_frames,
#                width*height*num_channels))
#    hidden = Variable(torch.zeros(1, n_hidden))
#    output, next_hidden = rnn(input[0], hidden)

