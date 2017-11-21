"""
Video Data Module.

This module reads video data and respective labels and stores it in numpy 
matrix.

@author: Gary Corcoran
@date: Nov. 20th, 2017

"""
import numpy as np
from sklearn import preprocessing
import cv2
from os.path import isfile

class VideoData():
    """ Class to hold and manipulate video data and labels. """
    def __init__(self, labels_path, num_videos, num_frames, width, height,
            num_channels):
        """
        Initialize Video Data class.

        @param  labels_path:    path to labels text file
        @param  num_videos:     number of videos to include in dataset
                                    @pre >= 1 and < 1750
        @param  num_frames:     number of frames to include in each video
                                    @pre >= 1 and < 100
        @param  width:          resize image to specified width
        @param  height:         resize image to specified height
        @param  num_channels:   number of color channels    
                                    @pre 1 or 3

        """
        self.path = labels_path
        self.num_videos = num_videos
        self.num_frames = num_frames
        self.width = width
        self.height = height
        self.num_channels = num_channels
        self.X = np.zeros((num_videos, num_frames, width*height*num_channels),
                dtype=np.uint8)
        self.y = np.zeros((num_videos), dtype=np.uint8)

    def read_video(self, video_path):
        """
        Read and store data from video.

        @param  video_path:     path of video file

        @return X:              numpy array of stored video
        """
        cap = cv2.VideoCapture(video_path)
        # array to store data for each video
        X_vid = np.zeros((self.num_frames,
                    self.height*self.width*self.num_channels), dtype=np.uint8)
        # for given number of frames
        for i in range(self.num_frames):
            _, frame = cap.read()
            frame = cv2.resize(frame, (self.width, self.height))
            if self.num_channels == 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = np.reshape(frame,
                    (self.width*self.height*self.num_channels))
            X_vid[i, :] = frame
        cap.release()
        return X_vid

    def read_data(self):
        """
        Read and store videos and labels located at labels_path.
    
        """
        if isfile('X_videos.npy') and isfile('y_videos.npy'):
            print('Loading dataset...')
            self.X = np.load('X_videos.npy')
            self.y = np.load('y_videos.npy')
        else:
            print('Creating dataset...')
            # create empty arrays for features and labels
            with open(self.path, 'r') as file:
                for i, line in enumerate(file):
                    print(i)
                    # break if reached video limit
                    if i == self.num_videos:
                        break
                    video_path, video_label = line.split()
                    self.X[i, :, :] = self.read_video(video_path)
                    self.y[i] = int(video_label)
            np.save('X_videos.npy', self.X)
            np.save('y_videos.npy', self.y)

    def normalize_data(self):
        """
        Normalize data to have a mean of 0 and variance of 1.

        """
        print('Normalizing dataset...')
        self.X = np.reshape(self.X, (self.num_videos*self.num_frames, -1))
        self.X = np.float64(self.X)
        self.X = preprocessing.scale(self.X, copy=False)
        self.X = np.reshape(self.X, (self.num_videos, self.num_frames, -1))
