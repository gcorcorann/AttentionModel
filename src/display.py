#!/usr/bin/env python3
"""
Display Dashcam Videos and Respective Attention Levels.

@author: Gary Corcoran
@date: Jan. 22nd, 2017

USAGE: python display.py
"""
import cv2
import numpy as np

def read_labels(labels_path):
    """
    Read video paths annotations and store in ndarry.

    @param  labels_path:    path to video annotations @pre string

    @return data:   ndarray holding video paths and annotations
    """
    with open(labels_path, 'r') as file:
        data = file.read()
        data = data.split()
        data = np.array(data)
        data = np.reshape(data, (-1, 2))
    return data

def display_sample(data):
    """
    Display sample videos.

    @param  data:   ndarray consisting of video paths and annotations
    """
    try:
        cap = cv2.VideoCapture()
        for row in data:
            vid_path, label = row
            cap.open(vid_path)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (224, 224))
                cv2.putText(frame, label, (10,30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0,0,255))
                cv2.imshow('Frame', frame)
                cv2.waitKey(100)
    except KeyboardInterrupt:
        cap.release()
        cv2.destroyAllWindows()

def main():
    """Main Function."""
    print(__doc__)
    labels_path = '../labels_gary.txt'
    data = read_labels(labels_path)
    
    display_sample(data)

if __name__ == '__main__':
    main()
