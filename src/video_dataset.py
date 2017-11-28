#!/usr/bin/env python3
"""
Video Dataset Module.

This module reads video dataset and respective labels and stores it in numpy 
matrix.

@author: Gary Corcoran
@date_created: Nov. 20th, 2017

USAGE:  python video_dataset.py [<labels_source>]

Keys:
    q   -   skip video
"""
import numpy as np
from sklearn import preprocessing
import cv2
from os.path import isfile
from video_player import Video
from optical_flow import OpticalFlow

class VideoDataset():
    """
    Class to hold and manipulate video data and labels.
    """
    def __init__(self, labels_path, num_videos, width, height, processor):
        """
        Initialize Video Dataset class.

        @param  labels_path:    path to labels text file
        @param  num_videos:     number of videos to include in dataset
                                    @pre >= 1 and < 1750
        @param  width:          resize frame to this width
        @param  height:         resize frame to this height
        @param  processor:      frame processor object
        """
        self._labels_path = labels_path
        self._num_videos = num_videos
        # initialize video reader object
        self._video = Video()
        self.set_video_params(width, height, processor)
        # store data, 99 frames from flow, and 2 dimensions, fx, fy
        self._X = np.zeros((num_videos, 99, height, width, 2), dtype=np.float32)
        self._y = np.zeros((num_videos), dtype=np.int32)

    def read_data(self):
        """
        Read and store videos and labels located at labels_path.

        @return X:  dataset features in format 
                        [num_videos, num_frames, num_feats]
        @return y:  dataset labels in format
                        [num_videos]
        """
        # create empty arrays for features and labels
        with open(self._labels_path, 'r') as file:
             for i, line in enumerate(file):
                # break if reached video limit
                if i == self._num_videos:
                    break
                print('Video', i+1, '/', self._num_videos)
                video_path, video_label = line.split()
                print(video_path, video_label)
                # run video
                self._video.set_video_path(video_path)
                # return processed frames (i.e. flow)
                X_video = self._video.run(return_frames=True)
                self._X[i] = X_video
                self._y[i] = video_label
                print()
        return self._X, self._y

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
    # optical flow parameters
    opt_params = {'pyr_scale': 0.5, 'levels': 3, 'winsize': 15, 'iterations': 3,
        'poly_n': 5, 'poly_sigma': 1.2}
    # create optical flow object
    opt = OpticalFlow(**opt_params)
    # create video data object
    vids = VideoDataset(labels_path, num_videos=10, width=100, height=100,
            processor=opt)
    X, y = vids.read_data()
    print(X.dtype)
    print(y.dtype)
    print('X:', X.shape)
    print('y:', y.shape)
    # save flow datast
    np.save('../data/X_flow.npy', X)
    np.save('../data/y.npy', y)

if __name__ == '__main__':
    main()
