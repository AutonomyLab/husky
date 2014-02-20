#! /usr/bin/env python

#import roslib, rospy
import sys, cv, cv2
#from std_msgs.msg import String
#from geometry_msgs.msg import Twist, Point, Point32, Polygon
#from sensor_msgs.msg import Image, LaserScan, Joy
import numpy as np
#from cv_bridge import CvBridge, CvBridgeError
import subprocess, signal
import math
import atexit
import os, time
import cPickle as pickle

from optparse import OptionParser


#--------------------------------------------------------------

parser = OptionParser()

parser.add_option("-f", "--frames", dest="frame_dir",
        help="directory to look for frame*.jpg")

parser.add_option("-o", "--output", dest="output_dir",
        help="directory to save annotated frames")

parser.add_option("-a", "--append", dest="append", default=False, action="store_true",
        help="append mode, for adding further annotations")

parser.add_option("-r", "--framerate", dest="framerate", default=-1, action="store", type="float",
        help="playback framerate in frames per second")

parser.add_option("-i", "--input-append", dest="inappend", help="dir to find previous annotations")

(options, args) = parser.parse_args()

#--------------------------------------------------------------

mouse_down = False
mouse_x = 0
mouse_y = 0

def on_mouse(event, x, y, flags, param):
    global mouse_down, mouse_x, mouse_y
    if event == cv2.EVENT_LBUTTONDOWN:
        # start gesture annotation at x,y, and current time
        mouse_down = True
        mouse_x = x
        mouse_y = y

        print "ANNOTATING"

    elif event == cv2.EVENT_LBUTTONUP:
        # stop gesture annotation that we had been recording
        mouse_down = False

        print "FINISHED ANNOTATION"

#--------------------------------------------------------------

interval = 1.0 / options.framerate

cv2.namedWindow("frame")

cv2.setMouseCallback("frame", on_mouse)

print "reading in image files"

start = 0
frame = start
dat = []
outpic = "%s%sannotations.pic" % (options.output_dir, os.sep)
inpic = "%s%sannotations.pic" % (options.inappend, os.sep)
old_ann = []
annotations = []
if options.append:
    # load the old annotations up so we can append annotations
    with open(inpic) as f:
        old_ann = pickle.load(f)


try:
    while True:

        filename = "%s%sframe%04d.jpg" % (options.frame_dir, os.sep, frame)
        outjpg = "%s%sframe%04d.jpg" % (options.output_dir, os.sep, frame)

        if not os.path.isfile(filename):
            print "done reading images in"
            break

        print "reading image: %s" % filename

        img = cv2.imread(filename)

        if not options.append:
            old_ann.append([])

        annotations.append([])

        dat.append((img, outjpg))
        frame += 1

except IOError, e:
    pass # we're done


finally:
    frame = 0
    print "annotation phase"

    for d in dat:
        img = d[0]
        outjpg = d[1]
        display_img = np.matrix.copy(img)
        annotations[frame].extend(old_ann[frame])

        if mouse_down:
            annotations[frame].append((mouse_x, mouse_y))

        for point in annotations[frame]:
            cv2.circle(display_img, point, 10, (0, 255, 0), 3)

        cv2.imshow("frame", display_img)

        frame += 1
        cv2.waitKey(20)

    cv2.destroyAllWindows()

    print "dumping annotations"
    with open(outpic, "w") as f:
        pickle.dump(annotations, f)


    print "done!"
