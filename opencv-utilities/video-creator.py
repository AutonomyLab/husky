#! /usr/bin/env python

import sys, cv, cv2, os
import numpy as np
import subprocess, signal
import math
import atexit
import cPickle as pickle
from sklearn.cluster import DBSCAN
from sklearn import metrics, preprocessing
import pymeanshift as pms
from optparse import OptionParser
import time

parser = OptionParser()

parser.add_option("-i", "--input", dest="input_dir", help="directory with frames")

parser.add_option("-s", "--start", dest="start_frame", default="0", help="frame to start on")

parser.add_option("-r", "--framerate", dest="framerate", default="30", help="playback rate")

parser.add_option("-c", "--crop-husky", dest="crop", action="store_true", default=False, help="crop out the image header from the Husky?")

parser.add_option("--save", dest="save", action="store", default=None, help="directory to save the rendered frames to")

(options, args) = parser.parse_args()





video = None

framerate = int(options.framerate)
frame = int(options.start_frame)
while True:
    framefile = "%s%sframe%04d.jpg" % (options.input_dir, os.sep, frame)
    print framefile

    if not os.path.isfile(framefile):
        print "done"
        break

    img = cv2.imread(framefile)
    if options.crop:
        img = img[20:, :]

    if video == None:
        vidfile = "%s%svideo.avi" % (options.save, os.sep)
        height, width, layers = img.shape
        video = cv2.VideoWriter(vidfile, cv2.cv.CV_FOURCC('M','J','P','G'), framerate, (width, height), True)

    video.write(img)

    frame += 1

cv2.destroyAllWindows()
