#!/usr/bin/env python3
"""
Video Dataset Module.

This module reads video dataset and respective labels and stores it in numpy 
matrix.

@author: Gary Corcoran
@date_created: Nov. 20th, 2017

USAGE:  python video_dataset.py [<labels_source>]

Keys:
    q   -   exit program
"""
import numpy as np
from sklearn import preprocessing
import cv2
from os.path import isfile
from video_player import Video

class VideoDataset():
    """
    Class to hold and manipulate video data and labels.
    """
    def __init__(self, labels_path, num_videos):
        """
        Initialize Video Dataset class.

        @param  labels_path:    path to labels text file
        @param  num_videos:     number of videos to include in dataset
                                    @pre >= 1 and < 1750
        """
        self._labels_path = labels_path
        self._num_videos = num_videos
        # initialize video reader object
        self._video = Video()

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
        with open(self._labels_path, 'r') as file:
             for i, line in enumerate(file):
                # break if reached video limit
                if i == self._num_videos:
                    break
                print('Video', i+1, '/', self._num_videos)
                video_path, video_label = line.split()
                print(video_path, video_label)
                print()
#                X.append(self._read_video(video_path))
#                y.append(int(video_label))
        return np.array(X), np.array(y)

    def set_video_params(self, width, height, processor=None):
        """
        Set parameters for video reader module.

        @param  width:  resized video width
        @param  height: resized video height
        @param  processor:  frame processor object
        """
        self._video.set_dimensions(width, height)
        self._video.set_processor(processor)

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
    # create video data object
    vids = VideoDataset(labels_path, num_videos=10)
    # set video reader parameters
    vids.set_video_params(width=300, hieght=200, processor=opt)
#    X, y = vids.read_data()
#    print('X:', X.shape)
#    print('y:', y.shape)

if __name__ == '__main__':
    main()
