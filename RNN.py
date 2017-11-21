################################################################################
#                           RNN Attention Model                                #
#                                                                              #
# @author: Gary Corcoran                                                       #
# @date: Nov. 18th, 2017                                                       #
#                                                                              #
################################################################################
import numpy as np
import glob
import cv2
import random
from sklearn import preprocessing
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

######################
## HELPER FUNCTIONS ##
######################
def read_data(labels_path, width, height):
    """
    Read and store videos and labels located at labels_path.

    @param  labels_path:    path to labels text file
    @param  width:  resize image to specified width
    @param  height: resize image to specified height

    @return X:  formatted data in form [instances, sequence_length, features]
    @return y:  formatted data in form [instances]
    """
    from os.path import isfile
    if isfile('X_videos.npy') and isfile('y_videos.npy'):
        print('Loading dataset...')
        X = np.load('X_videos.npy')
        y = np.load('y_videos.npy')
        return X, y
    # create dataset
    print('Creating dataset...')
    num_videos = 1750
    video_length = 100
    # create empty arrays for features and labels
    X = np.zeros((num_videos, video_length, width*height), dtype=np.uint8)
    y = np.zeros((num_videos), dtype=np.uint8)
    with open(labels_path, 'r') as file:
        for i, line in enumerate(file):
            print(i)
            video_path, video_label = line.split()
            X_vid = read_video(video_path, width, height)
            X[i, :, :] = X_vid
            y[i] = int(video_label)
    np.save('X_videos.npy', X)
    np.save('y_videos.npy', y)
    return X, y
            
def read_video(video_path, width, height):
    """
    Read and store data from video.

    @param  video_path: path of video file
    @param  width:  stored frame width
    @param  height: stored frame height

    @return X:  numpy array of stored features
    """
    cap = cv2.VideoCapture(video_path)
    # list for video data
    video_length = 100  # 100 frames per video
    # array to store video feature data
    X = np.zeros((video_length, height*width), dtype=np.uint8)
    for i in range(video_length):
        _, frame = cap.read()
        frame = cv2.resize(frame, (height,width))
        frame = np.reshape(frame, (height*width))
        X[i, :] = frame
    cap.release()
    return X

def sequence_to_tensor(X_vid):
    vid_length, num_feats = X_vid.shape
    tensor = torch.zeros(vid_length, 1, num_feats)
    return tensor

def category_from_output(output):
    top_n, top_i = output.data.topk(1)
    return top_i[0][0] + 1


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax()

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.h2o(hidden)
        output = self.softmax(output)
        return output, hidden


if __name__ == '__main__':
    labels_path = 'labels_gary.txt'
    width = height = 10
    X_train, y_train = read_data('labels_gary.txt', width, height)
    X = np.reshape(X_train, (X_train.shape[0]*X_train.shape[1], -1))
    X = np.float64(X)
    X_scaled = preprocessing.scale(X)
#    img_mean = np.uint8(np.mean(np.mean(X, axis=0), axis=0))
#    X -= img_mean
#    
#    for video in X:
#        for frame in video:
#            cv2.imshow("Frame", np.uint8(np.reshape(frame, (width,height,3))))
#            cv2.waitKey(1)

#    n_categories = 4
#    n_hidden = 128
#    rnn = RNN(3*width*height, n_hidden, n_categories)
#    print(rnn)
#    input = Variable(sequence_to_tensor(X[0]))
#    hidden = Variable(torch.zeros(1, n_hidden))
#    output, next_hidden = rnn(input[0], hidden)
#    print(output)
#    print(category_from_output(output))
