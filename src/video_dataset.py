#!/usr/bin/env python3
"""
Video Dataset Module.

This module reads video dataset and respective labels and stores it in numpy 
matrix.

@author: Gary Corcoran
@date_created: Nov. 20th, 2017

USAGE:  python video_dataset.py [<labels_source>]

Keys:
    q   -   skip video (only when video is being displayed)
"""
import numpy as np
import cv2
from video_player import Video

class VideoDataset():
    """
    Class to hold and manipulate video data and labels.
    """
    def __init__(self, labels_path):
        """
        Initialize Video Dataset class.

        @param  labels_path:    path to labels text file
        """
        self._labels_path = labels_path
        self._video = Video()

    def read_data(self):
        """
        Read and store videos and labels located at labels_path.

        @return X:  array of video path names
        @return y:  array of video labels
        """
        # create empty arrays for features and labels
        X = []
        y = []
        with open(self._labels_path, 'r') as file:
             for i, line in enumerate(file):
                video_path, video_label = line.split()
                X.append(video_path)
                y.append(video_label)
        return np.array(X), np.array(y)

    def partition_data(self, X, y, ratio):
        """
        Partition dataset.

        @param  X:  array of input image paths
        @param  y:  array of labels
        @param  ratio:  training/testing split

        @return X_tr:   training input image paths
        @return y_tr:   training image labels
        @return X_te:   testing input image paths
        @return y_te:   testing image labels
        """
        num_examples = X.shape[0]
        indices = np.random.permutation(num_examples)
        idx = int(ratio*num_examples)
        X_tr = X[indices[:idx]]
        y_tr = y[indices[:idx]]
        X_te = X[indices[idx:]]
        y_te = y[indices[idx:]]
        return X_tr, y_tr, X_te, y_te

    def set_video_params(self, width, height, processor=None):
        """
        Set parameters for video reader module.

        @param  width:  resized video width
        @param  height: resized video height
        @param  processor:  frame processor object
        """
        self._video.set_dimensions(width, height)
        self._video.set_processor(processor)

    def process_batch(self, X, y, batch_size):
        """
        Process mini batch.

        @param  X:              array of input video paths
        @param  y:              array of input video labels
        @param  batch_size:     size of mini-batch for SGD

        @return X_batch:        array of processed mini batch of videos
        @return y_batch:        array mini batch of video labels
        """
        print('Retrieving/processing mini-batch...')
        num_examples = X.shape[0]
        indices = np.random.permutation(num_examples)[:batch_size]
        X_batch = []
        y_batch = np.asarray(y[indices], np.int64)
        y_batch -= 1
        # for each video path
        for vid_path in X[indices]:
            self._video.set_video_path(vid_path)
            x_proc = self._video.run(display=False, return_frames=True)
            X_batch.append(x_proc)
        X_batch = np.array(X_batch)
        print('X_batch:', X_batch.shape)
        print('y_batch:', y_batch.shape)
        return X_batch, y_batch

def main():
    """ Main Function. """
    import sys
    from optical_flow import OpticalFlow
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
    vids = VideoDataset(labels_path)
    vids.set_video_params(width=300, height=200, processor=opt)
    X, y = vids.read_data()
    print('X:', len(X))
    print('y:', len(y))
    X_tr, y_tr, X_te, y_te = vids.partition_data(X, y, ratio=0.7)
    X_batch, y_batch = vids.process_batch(X_tr, y_tr, batch_size=10) 

if __name__ == '__main__':
    main()
