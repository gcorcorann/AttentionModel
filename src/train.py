#!/usr/bin/env python3
"""
C-RNN Training/Evaluation Module.

@author:        Gary Corcoran
@date_created:  Dec. 2nd, 2017

USAGE:  python train.py

KEYS:
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from optical_flow import OpticalFlow
from video_dataset import VideoDataset
from crnn import CRNN

class Trainer():
    """ Trains and Evaluates CRNN model. """
    def __init__(self, crnn, vids):
        self.crnn = crnn
        self.vids = vids
    
    def train(self, X_tr, y_tr, epochs, batch_size):
        print('Training Model...')
        criterion = nn.NLLLoss()
        optimizer = optim.SGD(self.crnn.parameters(), lr=0.0001, momentum=0.9)
        num_examples = len(X_tr)
        for epoch in range(epochs):
            print('Epoch:', epoch)
            running_loss = 0.0
            # pass through all training examples
            for i in range(num_examples // batch_size):
                # get processed batch of random training examples
                X_batch, y_batch = self.vids.process_batch(X_tr, y_tr, 
                        batch_size)
                # reshape into nSequences x nBatch x nChannels x Height x Width
                X_batch = np.swapaxes(X_batch, 0, 1)
                X_batch = np.moveaxis(X_batch, -1, 2)
                # wrap in pytorch variable
                X_var = Variable(torch.from_numpy(X_batch))
                y_var = Variable(torch.from_numpy(y_batch))
                # zero the parameter gradients
                # zero initial hidden state
                hidden = self.crnn.init_hidden(batch_size)
                # for each frame in sequence
                loss = 0
                for j in range(X_var.size()[0]):
                    # pass through CRNN
                    output, hidden = self.crnn(X_var[j], hidden)
                    loss += criterion(output, y_var)
                loss.backward()
                optimizer.step()
                running_loss += loss.data[0]

            # print statistics
            print('\tEpoch Loss:', running_loss / num_examples * batch_size)
            running_loss = 0

        print('Finished Training...')

def main():
    """ Main Function. """
    print(__doc__)
    # optical flow parameters
    opt_params = {'pyr_scale': 0.5, 'levels': 3, 'winsize': 15, 'iterations': 3,
        'poly_n': 5, 'poly_sigma': 1.2}
    # create optical flow object
    opt = OpticalFlow(**opt_params)
    # video dataset parameters
    labels_path = '../labels_gary.txt'
    width = 100
    height = 100
    processor = opt
    # create video data object
    vids = VideoDataset(labels_path)
    vids.set_video_params(width, height, processor)
    # read video paths and labels
    X, y = vids.read_data()
    # partition dataset
    X_tr, y_tr, X_te, y_te = vids.partition_data(X, y, ratio=0.0058)
    print(X_tr.shape)
    print(y_tr.shape)
    # create CRNN model
    crnn = CRNN()
    print(crnn)
    # train model
    tr = Trainer(crnn, vids)
    tr.train(X_tr, y_tr, epochs=5, batch_size=5)

if __name__ == '__main__':
    main()
