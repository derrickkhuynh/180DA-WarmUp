#modified from opencv-python example code at
# https://github.com/opencv/opencv/blob/master/samples/python/edge.py
# https://docs.opencv.org/3.4.13/da/d97/tutorial_threshold_inRange.html

# https://docs.opencv.org/3.4.13/d3/dc0/group__imgproc__shape.html#ga95f5b48d01abc7c2e0732db24689837b

# Majority of code is from the inRange example, parts were taken from edge.py to get the video capture, 
# and I used the last link to understand how to use the findContours function.


#to switch to RGB, I just changed the values 

#!/usr/bin/env python

'''
This sample demonstrates Canny edge detection.
Usage:
  edge.py [<video source>]
  Trackbars control edge thresholds.
'''

# Python 2/3 compatibility
from __future__ import print_function

import cv2
import numpy as np

# relative module
import video

# built-in module
import sys

max_value = 255
max_value_H = 360//2
low_H = 0
low_S = 0
low_V = 0
high_H = max_value_H
high_S = max_value
high_V = max_value
HSV_detection_name = 'HSV'
low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'


def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = val
    low_H = min(high_H-1, low_H)
    cv2.setTrackbarPos(low_H_name, HSV_detection_name, low_H)

def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    high_H = max(high_H, low_H+1)
    cv2.setTrackbarPos(high_H_name, HSV_detection_name, high_H)

def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    low_S = min(high_S-1, low_S)
    cv2.setTrackbarPos(low_S_name, HSV_detection_name, low_S)

def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    high_S = max(high_S, low_S+1)
    cv2.setTrackbarPos(high_S_name, HSV_detection_name, high_S)

def on_low_V_thresh_trackbar(val):
    global low_V
    global high_V
    low_V = val
    low_V = min(high_V-1, low_V)
    cv2.setTrackbarPos(low_V_name, HSV_detection_name, low_V)

def on_high_V_thresh_trackbar(val):
    global low_V
    global high_V
    high_V = val
    high_V = max(high_V, low_V+1)
    cv2.setTrackbarPos(high_V_name, HSV_detection_name, high_V)


def main():
    try:
        fn = sys.argv[1]
    except:
        fn = 0

    def nothing(*arg):
        pass

    cv2.namedWindow(HSV_detection_name)
    cv2.createTrackbar(low_H_name, HSV_detection_name, low_H, max_value_H, on_low_H_thresh_trackbar)
    cv2.createTrackbar(high_H_name, HSV_detection_name , high_H, max_value_H, on_high_H_thresh_trackbar)
    cv2.createTrackbar(low_S_name, HSV_detection_name , low_S, max_value, on_low_S_thresh_trackbar)
    cv2.createTrackbar(high_S_name, HSV_detection_name , high_S, max_value, on_high_S_thresh_trackbar)
    cv2.createTrackbar(low_V_name, HSV_detection_name , low_V, max_value, on_low_V_thresh_trackbar)
    cv2.createTrackbar(high_V_name, HSV_detection_name , high_V, max_value, on_high_V_thresh_trackbar)


    cap = video.create_capture(fn)
    while True:
        _flag, img = cap.read()
        #swap the commenting on the next 2 lines to switch to RGB instead of HSV.
        hsv = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV)
        #hsv = img.copy()
        hsv_mask = cv2.inRange(hsv, (low_H, low_S, low_V), (high_H, high_S, high_V))

        hsv_contours, hierarchy = cv2.findContours( hsv_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for c in hsv_contours:
            x, y, w, h = cv2.boundingRect(c)
            hsv_rect = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2) 
        
        cv2.imshow(HSV_detection_name, hsv_rect)
        ch = cv2.waitKey(5)
        if ch == 27:
            break

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv2.destroyAllWindows()