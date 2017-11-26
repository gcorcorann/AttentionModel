"""
RNN Training and Evaluation Module.

This module utilizes a predefined RNN model to train and evaluate the
performace.

@author: Gary Corcoran
@date: Nov. 21st, 2017

"""
import random
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


class Training():
    """ Trains RNN module. """
    def __init__(self, rnn, X_train, y_train, learning_rate, num_epochs):
        """
        Initialize training object.

        @param  rnn:    RNN class model
        @param  X_train:    numpy array of input training data in the form
                            [num_videos, num_frames, num_feats]
        @param  y_train:    numpy array of training labels in the form
                            [num_videos]
        @param  learning_rate:  gradient descent learning rate
        @param  num_epochs: number of training epochs
        """
        self.rnn = rnn
        self.Xtr = np.float32(X_train)
        self.ytr = np.uint8(y_train)
        self.lr = learning_rate
        self.num_epochs = num_epochs

    def __sequence_to_tensor(self, x):
        """
        Converts data sequence to tensor.

        @param  x:  numpy array consising of video data [num_frames, num_feats]

        @return x:  tensor of dimensions [num_frames, 1, num_feats]
        """
        # add dimension for batch size placeholder
        x = np.expand_dims(x, axis=1)
        return torch.from_numpy(x)

    def __random_training_example(self):
        """
        Get a random training example.

        @return x:          numpy array containing training instance
        @return x_rand:     pytorch Variable consisting of one training instance
        @return y:          integer containing training label
        @return y_rand:     pytorch Variable consisting of one training label
        """
        r = random.randint(0, self.Xtr.shape[0]-1)
        x = self.Xtr[r]
        x_rand = Variable(self.__sequence_to_tensor(x))
        # minus one since labels go from [1-4]
        y = int(self.ytr[r] - 1)
        y_rand = Variable(torch.LongTensor([y]))
        return x, x_rand, y, y_rand

    def __train_instance(self, x_var, y_var):
        """
        Train RNN model on training instance.

        @param  x_var:  pytorch Variable containing single training instance
        @param  y_var:  pytorch Variable containing single training label

        @return output: distribution of output labels
        @return loss:   scalar loss from training instance
        """
        hidden = self.rnn.init_hidden()
        self.rnn.zero_grad()
        criterion = nn.NLLLoss()
        # for each frame in training instance
        for frame in range(x_var.size()[0]):
            output, hidden = self.rnn(x_var[frame], hidden)
        loss = criterion(output, y_var)
        loss.backward()
        # add parameters' gradients to their values, multiplied by learning rate
        for p in self.rnn.parameters():
            p.data.add_(-self.lr, p.grad.data)
        return output, loss.data[0]

    def __label_from_output(self, output):
        """
        Retrieve label from output.

        @param  output: pytorch Variable containing output distribution of
                        labels
        
        @return label:  label associated with output Variable
        """
        _, top_i = output.data.topk(1)
        # add 1 to have range [1-4]
        label = top_i[0][0] + 1
        return label

    def train(self):
        """
        Tranin RNN.

        @return all_losses: list of average loss for each epoch
        """
        print('Training Model...')
        # keep track of losses
        current_loss = 0
        all_losses = []
        num_instances = self.Xtr.shape[0]
        for epoch in range(1, self.num_epochs+1):
            print('Epoch:', epoch)
            for i in range(0, num_instances):
                # get random training example
                x, x_var, y, y_var = self.__random_training_example()
                output, loss = self.__train_instance(x_var, y_var)
                current_loss += loss
            current_loss /= num_instances
            all_losses.append(current_loss)
            print('\tLoss:', current_loss)
            current_loss = 0
            print('\tSample Instance:')
            print('\tPredicted:', self.__label_from_output(output))
            print('\tActual:', y)
            print()
        return all_losses
