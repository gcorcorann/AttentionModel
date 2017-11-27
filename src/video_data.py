#!/usr/bin/env python3
"""
Video Data Module.

This module reads video data and respective labels and stores it in numpy 
matrix.

@author: Gary Corcoran
@date_created: Nov. 20th, 2017

USAGE:  python video_data.py [<labels_source>]

Keys:
    q   -   exit program
"""
import numpy as np
from sklearn import preprocessing
import cv2
from os.path import isfile

class VideoData():
    """
    Class to hold and manipulate video data and labels.
    """
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
        self._path = labels_path
        self._num_videos = num_videos
        self._num_frames = num_frames
        self._width = width
        self._height = height
        self._num_channels = num_channels
        self._cap = cv2.VideoCapture()

    def _read_video(self, video_path):
        """
        Read and store data from video.

        @param  video_path:     path of video file

        @return X:              numpy array of stored video
        """
        # list to store video data
        X_vid = []
        # open video
        if self._cap.open(video_path) is False:
            print('Could not open video.')
        # for given number of frames
        for i in range(self.num_frames):
            ret, frame = self._cap.read()
            if ret is False:
                print('Reached end of file.')
            frame = cv2.resize(frame, (self.width, self.height))
            if self.num_channels == 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            X_vid.append(frame)
        return X_vid

    def read_data(self):
        """
        Read and store videos and labels located at labels_path.

        @return X:  dataset features in format 
                        [num_videos, num_frames, num_feats]
        @return y:  dataset labels in format
                        [num_videos]
        """
        print('Loading dataset...')
        # list for datasets
        X = []
        y = []
        # create empty arrays for features and labels
        with open(self.path, 'r') as file:
             for i, line in enumerate(file):
                # break if reached video limit
                if i == self.num_videos:
                    break
                print('Video', i+1, '/', self.num_videos)
                video_path, video_label = line.split()
                print(video_path, video_label)
                print()
                X.append(self._read_video(video_path))
                y.append(int(video_label))
        return np.array(X), np.array(y)

def main():
    """ Main Function. """
    import sys
    print(__doc__)
    if len(sys.argv) >= 2:
        # set command line input to labels path
        labels_path = sys.argv[1]
    else:
        # set default labels path
        labels_path = '../labels_gary.txt'
    # initialize video dataset parameters
    vid_params = {'num_videos': 10, 'num_frames': 100, 'width': 300, 
        'height': 200, 'num_channels': 3}
    # create video data object
    vids = VideoData(labels_path, **vid_params)
#    X, y = vids.read_data()
#    print('X:', X.shape)
#    print('y:', y.shape)

if __name__ == '__main__':
    main()
