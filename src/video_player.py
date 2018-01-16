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
        self._video_path = video_path
        self._processor = processor
        self._cap = None
        self._width = None
        self._height = None

    def __del__(self):
        """
        Object destructor to release resources.
        """
        if self._cap is not None:
            self._cap.release()
        cv2.destroyAllWindows()

    def set_video_path(self, video_path):
        """
        Set path to input video file.

        @param  video_path: path to input video file

        @modifies   self.video_path:    stores video file
        """
        self._video_path = video_path

    def get_video_path(self):
        """
        Returns video path.

        @return video_path: path to input video file
        """
        return self._video_path

    def _check_video_path(self):
        """
        Check if video path is set.

        @return ret:    true if video path is set, else false
        """
        if self.get_video_path() is None:
            print('VideoError: Please input video path before running.')
            return False
        return True

    def set_processor(self, processor):
        """
        Set frame processor.

        @param  processor:  frame processor object
        
        @modifies   self._processor:    stores object
        """
        self._processor = processor

    def check_processor(self):
        """
        Check if frame processor is set.

        @return ret:    true if frame processor is set, else false
        """
        if self._processor is None:
            return False
        return True

    def set_dimensions(self, width, height):
        """
        Set frame dimensions.

        @param  width:  resized frame width
        @param  height: resized frame height

        @modifies   self._width:    stores frame width
        @modifies   self._height:   stores frame height
        """
        self._width = width
        self._height = height

    def check_dimensions(self):
        """
        Check if frame dimensions were set.

        @return ret:    true if frame dimensions were set, else false
        """
        if self._width is not None and self._height is not None:
            return True
        return False

    def _is_opened(self):
        """
        Check if video is opened.

        @return ret:    true if video is opened, else false
        """
        ret = self._cap.isOpened()
        if ret is False:
            print('VideoError: Could not opened video file stored at:', 
                    self._video_path)
        return ret

    def _read(self):
        """
        Read video frame.

        @return ret:    false end of file, else true
        @return frame:  video frame
        """
        ret, frame = self._cap.read()
        return ret, frame

    def _display(self, frame):
        """
        Display video frame.
        
        @return ret:    false if user quit video, else true
        """
        cv2.imshow('Frame', frame)
        key = cv2.waitKey(33)
        # if user wants to exit
        if key == ord('q'):
            print('Video: User quit video.')
            return False
        return True

    def run(self, display=True, return_frames=False):
        """
        Play video stored in video_path.

        @param  return_frames:  if true and a processor is set, return array 
                                of processed frames
        @param  display:        if true display processed frames

        @return X_video:    array of processed frames
        """
        # check if user set video path
        if not self._check_video_path():
            return
        self._cap = cv2.VideoCapture(self._video_path)
        # check if video openec successfully
        if not self._is_opened():
            return
#        # read video frame
#        ret, frame1 = self._read()
#        # check if frame was successfully read
#        if not ret:
#            return
#        if not self.check_dimensions():
#            frame1 = cv2.resize(frame1, None, fx=0.5, fy=0.5)
#        else:
#            frame1 = cv2.resize(frame1, (self._width, self._height))
#        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        # initialize list of processed frames
        if return_frames:
            X_video = []
        # while video is still opened
        while self._is_opened():
            ret, frame2 = self._read()
            if not ret:
                break
            if not self.check_dimensions():
                frame2 = cv2.resize(frame2, None, fx=0.5, fy=0.5)
            else:
                frame2 = cv2.resize(frame2, (self._width, self._height))
            X_video.append(frame2)
#            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
#            # check if processor is set 
#            if self.check_processor():
#                flow = self._processor.compute(gray1, gray2)
#                # append processed frames into list
#                if return_frames:
#                    X_video.append(flow)
#                if display:
#                    disp_img = self._processor.visualize(frame2)
#            else:
#                if display:
#                    disp_img = frame2
#            # display
#            if display:
#                if not self._display(disp_img):
#                    break
#            # set previous frame to current frame
##            gray1 = gray2
#        if self.check_processor and return_frames:
        return np.array(X_video)

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
    vod.set_dimensions(width=300, height=200)
    X_video = vod.run(display=True, return_frames=True)
    print('X data:', X_video.shape)

if __name__ == '__main__':
    main()
