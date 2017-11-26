#!/usr/bin/env python3
"""
Optical Flow Module.

@author: Gary Corcoran
@date_created: Nov. 25th, 2017

USAGE: optical_flow.py [<video_source>]

Keys:
    any key - exit
"""
import numpy as np
import cv2

class OpticalFlow():
    """
    Optical Flow.
    """
    def __init__(self, pyr_scale, levels, winsize, iterations, poly_n,
            poly_sigma):
        """
        Initialize parameters.
        """
        self._pyr_scale = pyr_scale
        self._levels = levels
        self._winsize = winsize
        self._iterations = iterations
        self._poly_n = poly_n
        self._poly_sigma = poly_sigma
        self.flow = None

    def compute(self, gray1, gray2):
        """
        Compute optical flow.

        @param  gray1: grayscale initial frame
        @param  gray2: grayscale second frame

        @return flow:   optical flow between gray1 and gray2
        """
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None,
                self._pyr_scale, self._levels, self._winsize, self._iterations,
                self._poly_n, self._poly_sigma, 0)
        self.flow = flow
        return flow

    def _draw_flow(self, img, step):
        """
        Draw optical flow.

        @param  img:    input image to draw flow vectors on
        @param  step:   number of steps between flow vectors

        @return flow_img:   optical flow image
        """
        flow = self.flow
        h, w = flow.shape[:2]
        # copy original image to place flow vectors on
        flow_img = img.copy()
        y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
        fx, fy = flow[y, x].T
        lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines+0.5)
        cv2.polylines(flow_img, lines, 0, (0,255,0))
        for (x1,y1), (x2,y2) in lines:
            cv2.circle(flow_img, (x1,x1), 1, (0,255,0), -1)
        return flow_img

    def _draw_hsv(self):
        """
        Retrieve image to display flow.

        @return flow_img:   optical flow image
        """
        flow = self.flow
        h, w = flow.shape[:2]
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        # set constant saturation level
        hsv[:,:,1] = 255
        # compute magnitude and phase
        mag, ang = cv2.cartToPolar(flow[:,:,0], flow[:,:,1])
        hsv[:,:,0] = ang * 180 / np.pi / 2
        hsv[:,:,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        flow_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return flow_img
    
    def visualize(self, frame, step=16):
        """
        Visualize optical flow.

        @param  frame:  input frame to display vectors on
        @param  step:   distance between each flow vector

        @return flow_img:   image displaying optical flow
        """
        flow_hsv = self._draw_hsv()
        flow_vec = self._draw_flow(frame, step)
        h, w = self.flow.shape[:2]
        flow_img = np.zeros((h, w*2, 3), np.uint8)
        flow_img[:, :w] = flow_vec
        flow_img[:, w:] = flow_hsv
        return flow_img

def main():
    """ Testing function. """
    print(__doc__)
    # read sequence of images
    frame1 = cv2.imread('/home/gary/opencv/samples/data/basketball1.png')
    frame2 = cv2.imread('/home/gary/opencv/samples/data/basketball2.png')
    # convert to grayscale for flow
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    # optical flow params
    opt_params = {'pyr_scale': 0.5, 'levels': 3, 'winsize': 15,
        'iterations': 3, 'poly_n': 5, 'poly_sigma': 1.2}
    # create optical flow object
    opt = OpticalFlow(**opt_params)
    # compute and display flow
    flow = opt.compute(gray1, gray2)
    flow_img = opt.visualize(frame2, step=16)
    cv2.imshow('Optical Flow', flow_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
