#! /usr/bin/env python

# This node uses Fourier analysis to try to detect regions of the image
# with periodicity in a certain frequency range.

# Image regions of interest are narrowed down by looking for
# regions that contain motion... background subtraction

import cv, cv2
import numpy as np, matplotlib.pyplot as plot
import sys, os, cPickle as pickle
import rospy, roslib
from geometry_msgs.msg import Polygon, Point32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import math

#-----------------------------------------------------------------

class periodic_gestures:
    def __init__(self):
        rospy.init_node("periodic_gestures")

        self.cv_bridge = CvBridge()

        self.gesture_pub = rospy.Publisher(rospy.get_param("~gesture_topic", "periodic_gestures/gestures"), Polygon)
        self.viz_pub = rospy.Publisher(rospy.get_param("~viz_topic", "periodic_gestures/viz"), Image)

        rospy.Subscriber(rospy.get_param("~motion_topic", "motion_detection/motion"), Polygon, self.handle_motion)

        rospy.Subscriber(rospy.get_param("~image_topic", "axis/image_raw/decompressed"), Image, self.handle_image)

        # window of frames over which to do Fourier analysis
        self.TEMPORAL_WINDOW = rospy.get_param("~temporal_window", 120)
        # self.FREQ_COMPONENTS = [3,4,5,6,7]

        self.MIN_GESTURE_FREQUENCY = rospy.get_param("~min_gesture_freq", 0.5)
        self.MAX_GESTURE_FREQUENCY = rospy.get_param("~max_gesture_freq", 2)
        self.CAMERA_FRAMERATE = rospy.get_param("~camera_framerate", 30)

        # calculate the spectral bins using
        # the desired frequency range, the camera framerate
        # and the size of the temporal window
        bin_width = self.CAMERA_FRAMERATE / float(self.TEMPORAL_WINDOW)
        smallest_bin = int(self.MIN_GESTURE_FREQUENCY / bin_width) + 1
        freq_range = self.MAX_GESTURE_FREQUENCY - self.MIN_GESTURE_FREQUENCY
        largest_bin = smallest_bin + int(freq_range / bin_width)

        # these are our frequency bins of interest
        self.FREQ_COMPONENTS = range(smallest_bin, largest_bin+1)
        print "INIT: spectral bins of interest: %s" % self.FREQ_COMPONENTS

        # how picky are we about what constitutes a significant
        # peak in the frequency domain
        self.PEAK_STDEVS = rospy.get_param("~peak_sensitivity", 9)

        # peaks must be at least this high to qualify
        self.MIN_FREQ_INTENSITY = rospy.get_param("~min_peak", 50)

        # size of the overlapping subregion windows
        self.SPATIAL_WINDOW_X = rospy.get_param("~spatial_window_x", 10)
        self.SPATIAL_WINDOW_Y = rospy.get_param("~spatial_window_y", 10)

        # overlap factor of 2 means each window overlaps the previous by half
        self.OVERLAP_FACTOR = rospy.get_param("~overlap_factor", 2)

        # filled in as we receive images
        self.WIDTH = 0
        self.HEIGHT = 0
        self.DEPTH = 0

        # areas of significant motion, as received
        # from the motion_detection node
        self.motion_areas = None

        # where we are in the current temporal window
        self.temporal_window_index = 0

        # don't start looking for periodicity until we've filled
        # a whole temporal window
        self.temporal_window_full = False

        # the overlapping windows in the image to check
        self.spatial_windows = None

#-----------------------------------------------------------------

    def handle_motion(self, data):
        # unpack rectangles from super-polygon
        # and set self.motion_windows to be the list of rectangles

        data = data.points

        i = 0
        motion = []
        while i+2 < len(data):
            top_left = data[i]
            bottom_right = data[i+2]
            motion.append((top_left, bottom_right))
            i += 4

        self.motion_areas = motion

#-----------------------------------------------------------------

    def handle_image(self, data):
        if self.no_one_listening():
            # reset detection process so we start from
            # scratch once someone subscribes
            self.temporal_window_index = 0
            self.temporal_window_full = False
            return

        self.image = np.asarray(cv.GetMat(self.image))
        self.image = self.cv_bridge.imgmsg_to_cv(data, desired_encoding="mono8")

        # wait for motion detection first
        if self.motion_areas == None:
            return

        self.HEIGHT, self.WIDTH, self.DEPTH = self.image.shape

        # if we are at the beginning of a round
        if self.temporal_window_index == 0:
            # if we've filled a temporal window, then
            # run Fourier analysis on all the data we've accumulated
            if self.temporal_window_full:
                self.identify_periodic_windows()

            # determine the interesting regions for the upcoming round
            self.determine_windows_to_check()


        # get the average pixel intensity over each
        # of the windows of interest for this round
        self.average_each_subwindow()

        # end, increment window index with rollover
        self.temporal_window_index += 1
        if self.temporal_window_index == self.TEMPORAL_WINDOW:
            self.temporal_window_full = True
        self.temporal_window_index %= self.TEMPORAL_WINDOW

#-----------------------------------------------------------------

    def determine_windows_to_check(self):
        # run through self.motion_areas and construct a 
        # list of overlapping regions to check for periodicity

        if self.motion_areas == None:
            return

        # keep track of the areas we monitor
        # so we don't track the same rectangle twice
        flagged_areas = np.zeros((self.WIDTH, self.HEIGHT))

        self.spatial_windows = []

        for area in self.motion_areas:
            top_left = area[0]
            bottom_right = area[1]

            x = top_left[0]
            y = top_left[1]

            while x <= bottom_right[0] and y <= bottom_right[1]:
                if flagged_areas[x,y] == 0:
                    # add rectangle to regions of interest

                    window = ((x,y,
                        x+self.SPATIAL_WINDOW_X,
                        y+SPATIAL_WINDOW_Y),
                        [], # average pixel level over the temporal window
                        False) # whether periodic motion was detected

                    self.spatial_windows.append(window)

                    flagged_areas[x,y] = 1

                x += (self.SPATIAL_WINDOW_X / self.OVERLAP_FACTOR)
                y += (self.SPATIAL_WINDOW_Y / self.OVERLAP_FACTOR)

#-----------------------------------------------------------------
    
    def identify_periodic_windows(self):
        # run Fourier analysis over all windows we've
        # been accumulating data from, and publish windows that
        # contain periodicity, perhaps by clustering first

        if not self.TEMPORAL_WINDOW_FULL:
            return 

        motion_detected_windows = []

        for window in self.spatial_windows:
            time_domain = np.asarray(window[1])
            frequency_domain = abs(np.fft.fft(time_domain))

            avg = np.sum(frequency_domain[1:len(frequency_domain)/2]) / (float(len(frequency_domain)/2-1))

            std = np.std(frequency_domain[1:len(frequency_domain)/2])

            threshold = avg + std*self.PEAK_STDEVS

            periodic = False
            for f in self.FREQ_COMPONENTS:
                if (frequency_domain[f] >= threshold and
                        frequency_domain[f] > frequency_domain[1]):
                    periodic = True

            if periodic:
                motion_detected_windows.append(window)

        # run through motion_detected_windows and publish
        # a super-polygon the represents the locations where
        # periodic motion in the correct frequency range was found
        super_polygon = []
        for window in motion_detected_windows:
            p0 = (window[0][0], window[0][1])
            p1 = (p0[0] + self.SPATIAL_WINDOW_X, p0[1])
            p2 = (window[0][2], window[0][3])
            p3 = (p0[0], p0[1] + self.SPATIAL_WINDOW_Y)

            super_polygon.append(Point32(x=p0[0], y=p0[1], z=0))
            super_polygon.append(Point32(x=p1[0], y=p1[1], z=0))
            super_polygon.append(Point32(x=p2[0], y=p2[1], z=0))
            super_polygon.append(Point32(x=p3[0], y=p3[1], z=0))

        self.gesture_pub.publish(super_polygon)

        if self.viz_pub.get_num_connections() > 0:
            self.publish_viz(motion_detected_windows, self.image)
        
        # TODO: cluster these positive rectangle somehow so we can
        # have multiple different periodic motions detected in the
        # same scene

#-----------------------------------------------------------------

    def publish_viz(self, rectangles, img):
        for rect in rectangles:
            cv2.rectangle(img, rect[0][0], rect[0][2], (255, 255, 255))

        img = cv.fromarray(img)
        msg = self.cv_bridge.cv_to_imgmsg(img)
        self.viz_pub.publish(msg)

#-----------------------------------------------------------------

    def no_one_listening(self):
        return (self.gesture_pub.get_num_connections() > 1 and
                self.viz_pub.get_num_connections() > 1)

#-----------------------------------------------------------------

    def average_each_subwindow(self):
        # get an average of the pixels in each window
        # of interest that we can use at the end of the round,
        # or at each frame of the round once the temporal window
        # is full

        for window in self.spatial_windows:
            ymin = window[0][1]
            ymax = window[0][3]+1

            xmin = window[0][0]
            xmax = window[0][2]+1

            subimage = self.image[ymin:ymax, xmin:xmax]
            avg = np.sum(subimage) / (self.SPATIAL_WINDOW_X*self.SPATIAL_WINDOW_Y)

            window[1].append(avg)

#-----------------------------------------------------------------
# END CLASS PERIODIC_GESTURES
#-----------------------------------------------------------------

if __name__ == "__main__":
    pg = periodic_gestures()
    rospy.spin()
