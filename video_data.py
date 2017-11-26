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
            num_channels, ratio):
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
        @param  ration:         ratio of training examples
                                    @pre [0-1]
        """
        self.path = labels_path
        self.num_videos = num_videos
        self.num_frames = num_frames
        self.width = width
        self.height = height
        self.num_channels = num_channels
        self.ratio = ratio
        self.__Xtr = np.zeros((num_videos, num_frames, 
                    width*height*num_channels), dtype=np.uint8)
        self.__ytr = np.zeros((num_videos), dtype=np.uint8)

    def __read_video(self, video_path):
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

    def read_data(self, normalize, load):
        """
        Read and store videos and labels located at labels_path.

        @param  normalize:  if True normalize training and testing data
                                @pre True or False
        @param  load:       if True load dataset from disk
                                @pre True of False

        @return X_train:    training dataset features in format 
                            [num_videos, num_frames, num_feats]
        @return y_train:    training dataset labels in format
                            [num_videos]
        @return X_test:     testing dataset features, same format as X_train
        @return y_test:     testing dataset labels, same format at y_train
        """
        if load is True: 
            print('Loading dataset...')
            self.__Xtr = np.load('X_videos.npy')
            self.__ytr = np.load('y_videos.npy')
        else:
            print('Creating dataset...')
            # create empty arrays for features and labels
            with open(self.path, 'r') as file:
                for i, line in enumerate(file):
                    print('video', i, '/', self.num_videos)
                    # break if reached video limit
                    if i == self.num_videos:
                        break
                    video_path, video_label = line.split()
                    self.__Xtr[i, :, :] = self.__read_video(video_path)
                    self.__ytr[i] = int(video_label)
            np.save('X_videos.npy', self.__Xtr)
            np.save('y_videos.npy', self.__ytr)
        # split into training and testing set
        self.split_data()
        if normalize is True:
            self.normalize_data()
        return self.__Xtr, self.__ytr, self.__Xte, self.__yte

    def split_data(self):
        """
        Split data into training and testing sets.
        """
        indices = np.random.permutation(self.num_videos)
        idx = int(self.num_videos * self.ratio)
        self.__Xte = self.__Xtr[idx:]
        self.__yte = self.__ytr[idx:]
        self.__Xtr = self.__Xtr[:idx]
        self.__ytr = self.__ytr[:idx]
        
    def normalize_data(self):
        """
        Normalize data to have a mean of 0 and variance of 1.
        """
        print('Normalizing dataset...')
        # get number of videos in traning and testing datasets
        num_train = self.__Xtr.shape[0]
        num_test = self.__Xte.shape[0]
        # reshape into 2 dimensional array
        self.__Xtr = np.reshape(self.__Xtr, 
                (num_train * self.num_frames, -1))
        self.__Xte = np.reshape(self.__Xte, 
                (num_test * self.num_frames, -1))
        self.__Xtr = np.float64(self.__Xtr)
        self.__Xte = np.float64(self.__Xte)
        # perform normalization
        scalar = preprocessing.StandardScaler().fit(self.__Xtr)
        self.__Xtr = scalar.transform(self.__Xtr)
        self.__Xte = scalar.transform(self.__Xte)
        # reshape back into 3 dimensions
        self.__Xtr = np.reshape(self.__Xtr, 
                (num_train, self.num_frames, -1))
        self.__Xte = np.reshape(self.__Xte, 
                (num_test, self.num_frames, -1))
