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

        rospy.Subscriber(rospy.get_param("~motion_topic", "motion_detection"), Polygon, self.handle_motion)

        rospy.Subscriber(rospy.get_param("~image_topic", "axis/image_raw/decompressed"), Image, self.handle_image)

        self.gesture_pub = rospy.Publisher(rospy.get_param("~gesture_topic", "periodic_gestures"), Polygon)

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
        if self.gesture_pub.get_num_connections() < 1:
            # reset detection process so we start from
            # scratch once someone subscribes
            self.temporal_window_index = 0
            self.temporal_window_full = False
            return

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

            super_polygon.append(Point32(p0))
            super_polygon.append(Point32(p1))
            super_polygon.append(Point32(p2))
            super_polygon.append(Point32(p3))

        self.gesture_pub.publish(super_polygon)
        
        # TODO: cluster these positive rectangle somehow so we can
        # have multiple different periodic motions detected in the
        # same scene

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

        print "calculating average of each subwindow"
        for j in range(0, len(windows)):
            windows[j] = windows[j][1:]
            windows[j].append(avg_subwindow(image, j))
            if (DISPLAY_WAVEFORM and j == MONITOR_BOX):
                WAVEFORM = windows[j]

        TEMPORAL_WINDOW_FULL = (i - START_FRAME > TEMPORAL_WINDOW)

        print "calculating Fourier transforms."
        motion_detected_windows = identify_periodic_windows(windows)

        print "wrapping rectangles around areas of periodic motion"
        CURRENT_PERIODIC_RECTANGLES = wrap_rectangles_around_windows()
        PERIODIC_RECTANGLES.extend(CURRENT_PERIODIC_RECTANGLES)

#-----------------------------------------------------------------

def wrap_rectangles_around_windows():
    global PERIODIC_RECTANGLE_INDEX

    if len(motion_detected_windows) == 0:
        return []

    # just create a single bounding rectangle for now
    minx = sys.maxint
    miny = sys.maxint
    maxx = 0
    maxy = 0

    for window in motion_detected_windows:
        if (window[0] < minx):
            minx = window[0]
        if (window[1] < miny):
            miny = window[1]
        if (window[0] + SPATIAL_WINDOW_X > maxx):
            maxx = window[0] + SPATIAL_WINDOW_X
        if (window[1] + SPATIAL_WINDOW_Y > maxy):
            maxy = window[1] + SPATIAL_WINDOW_Y

    clamped_minx = max(minx-1, 0)
    clamped_miny = max(miny-1, 0)
    clamped_maxx = min(maxx, WIDTH - 1)
    clamped_maxy = min(maxy, HEIGHT - 1)

    # make the window square, for scaling
    if clamped_maxx - clamped_minx > clamped_maxy - clamped_miny:
        diff = (clamped_maxx - clamped_minx) - (clamped_maxy - clamped_miny)
        diffup = diff/2
        diffdown = diff - diffup

        clamped_miny -= diffup
        clamped_maxy += diffup

        if (clamped_miny < 0):
            clamped_maxy -= clamped_miny
            clamped_miny = 0
        elif (clamped_maxy > HEIGHT - 1):
            clamped_miny -= (clamped_maxy - HEIGHT - 1)
            clamped_maxy = HEIGHT - 1
    elif clamped_maxx - clamped_minx < clamped_maxy - clamped_miny:
        diff = (clamped_maxy - clamped_miny) - (clamped_maxx - clamped_minx)
        diffleft = diff/2
        diffright = diff - diffleft

        clamped_minx -= diffleft
        clamped_maxx += diffright

        if (clamped_minx < 0):
            clamped_maxx -= clamped_minx
            clamped_minx = 0
        elif (clamped_maxx > WIDTH - 1):
            clamped_minx -= (clamped_maxx - WIDTH - 1)
            clamped_maxx = WIDTH - 1

    # clean up remaining small differences...
    while clamped_maxx - clamped_minx > clamped_maxy - clamped_miny:
        if clamped_maxy < HEIGHT - 1:
            clamped_maxy += 1
        else:
            clamped_miny -= 1
    while clamped_maxx - clamped_minx < clamped_maxy - clamped_miny:
        if clamped_maxx < WIDTH - 1:
            clamped_maxx += 1
        else:
            clamped_minx -= 1

    # since we've added a new rectangle, increment the label
    PERIODIC_RECTANGLE_INDEX += 1

    return [((clamped_minx, clamped_miny), (clamped_maxx, clamped_maxy), 0, PERIODIC_RECTANGLE_INDEX)]


#-----------------------------------------------------------------

def identify_periodic_windows(windows):
    global FFT, FFT_THRESHOLD, SECOND_DEGREE_FFT

    window_locations = []

    # don't register windows until we've got a whole temporal window
    if not TEMPORAL_WINDOW_FULL:
        return window_locations

    # if we just want to run through the video real quick
    if SCAN:
        return window_locations

    for i in range(0, len(windows)):
        fft = abs(np.fft.fft(windows[i]))

        avg = 0
        for j in range(1, len(fft) / 2):
            avg += fft[j]
        avg /= len(fft) / 2 - 1

        fft_threshold = 0		
        if SPIKE_DETECTION == "factor":
            fft_threshold = avg * PEAK_MAGNITUDE_FACTOR
        elif SPIKE_DETECTION == "offset":
            fft_threshold = avg + PEAK_MAGNITUDE_OFFSET
        elif SPIKE_DETECTION == "combo":
            fft_threshold = avg * PEAK_MAGNITUDE_FACTOR + PEAK_MAGNITUDE_OFFSET
        elif SPIKE_DETECTION == "max":
            fft_max = max(fft[2:len(fft)/2])
            fft_threshold = fft_max
        elif SPIKE_DETECTION == "outlier":
            stdev = 0
            for j in range(1, len(fft) / 2):
                stdev += abs(avg - fft[j])
            stdev /= len(fft) / 2 - 1
            fft_threshold = avg + PEAK_STDEVS * stdev

        # must be greater than some baseline to rule out noise
        fft_threshold = max(fft_threshold, MIN_FREQ_INTENSITY)

        if i == MONITOR_BOX:
            FFT_THRESHOLD = fft_threshold

        if (DISPLAY_HIST and i == MONITOR_BOX):
            FFT = fft[0:len(fft)/2]
            FFT[0] = 0

        if (DISPLAY_SECOND_DEGREE_FOURIER and i == MONITOR_BOX):
            SECOND_DEGREE_FFT = abs(np.fft.fft(fft[1:len(fft)/2]))

        periodic = False
        for f in FREQ_COMPONENTS:	
            if (fft[f] >= fft_threshold and fft[f] > fft[1]):
                periodic = True
        if periodic:
            window_locations.append(get_subwindow_location(i))

    return window_locations


#-----------------------------------------------------------------

def get_subwindow_index(x, y):
    x = max(0, x-SPATIAL_WINDOW_X/2)
    y = max(0, y-SPATIAL_WINDOW_Y/2)

    x_index = x / (SPATIAL_WINDOW_X / OVERLAP_FACTOR)
    y_index = y / (SPATIAL_WINDOW_Y / OVERLAP_FACTOR)
    windows_per_row = WIDTH / (SPATIAL_WINDOW_X / OVERLAP_FACTOR)

    return x_index + y_index * windows_per_row


#-----------------------------------------------------------------


def get_subwindow_location(i):
    global SPATIAL_WINDOW_X, SPATIAL_WINDOW_Y, OVERLAP_FACTOR

    x = (i % ( WIDTH / (SPATIAL_WINDOW_X / OVERLAP_FACTOR) ) ) * SPATIAL_WINDOW_X / OVERLAP_FACTOR

    y = ( (i*SPATIAL_WINDOW_X / OVERLAP_FACTOR) / WIDTH ) * SPATIAL_WINDOW_Y / OVERLAP_FACTOR

    return (x, y)


#-----------------------------------------------------------------

def avg_subwindow(image, subwindow_index):
    # this function should compute the average pixel intensity
    # in a given subwindow.

    # for now, we'll try just a simple average,
    # but Junaed's paper seems to talk about Gaussian
    # weighted averages.

    # if we just want to run through the video real quick
    if (SCAN):
        return 0

    (firstX, firstY) = get_subwindow_location(subwindow_index)

    total = 0
    count = 1
    for x in range(firstX, firstX + SPATIAL_WINDOW_X - 1):
        for y in range(firstY, firstY + SPATIAL_WINDOW_Y - 1):
            # make sure our windows don't go outside the image
            if (x < WIDTH and y < HEIGHT):
                pixval = cv.Get2D(image, y, x)
                total += pixval[0]
                count += 1

    return total / count


#-----------------------------------------------------------------

if __name__ == "__main__":
    pg = periodic_gestures()
