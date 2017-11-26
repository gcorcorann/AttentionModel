#!/usr/bin/env python3
"""
Video Player Module.

@author: Gary Corcoran
@date_created: Nov. 24th, 2017

USAGE: python video.py [<video_source>]

Keys:
    q   -   exit video
"""
import numpy as np
import cv2

class Video():
    """ Video player. """
    def __init__(self, video_path=None, processor=None):
        """
        Initialize parameters.

        @param  video_path: path to input video file
        @param  processor:  frame processor object
        """
        self.video_path = video_path
        self._processor = processor

    def run(self):
        """
        Play video stored in video_path.
        """
        processor = self._processor
        cap = cv2.VideoCapture(self.video_path)
        ret, frame1 = cap.read()
        frame1 = cv2.resize(frame1, None, fx=0.5, fy=0.5)
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        while True:
            ret, frame2 = cap.read()
            frame2 = cv2.resize(frame2, None, fx=0.5, fy=0.5)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            # compute flow
            flow = processor.compute(gray1, gray2)
            flow_img = processor.visualize(frame2)
            cv2.imshow('Frame', flow_img)
            key = cv2.waitKey(30)
            if key == ord('q'):
                break
            gray1 = gray2
        cap.release()
        cv2.destroyAllWindows()


       

def main():
    """ Main Function. """
    import sys
    from optical_flow import OpticalFlow
    print(__doc__)
    if len(sys.argv) >= 2:
        # set command line input to video path
        video_path = sys.argv[1]
    else:
        # set default video path
        video_path = '/home/gary/datasets/accv/positive/000550.mp4'
    # optical flow params
    opt_params = {'pyr_scale': 0.5, 'levels': 3, 'winsize': 15,
        'iterations': 3, 'poly_n': 5, 'poly_sigma': 1.2}
    # create optical flow object
    opt = OpticalFlow(**opt_params)
    # create video player object
    vod = Video(video_path=video_path, processor=opt)
    vod.run()

if __name__ == '__main__':
    main()
