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

parser.add_option("-m", "--mode", dest="mode", default="1", help="image processing mode")

parser.add_option("-r", "--framerate", dest="framerate", default="30", help="playback rate")

parser.add_option("-l", "--list", dest="list", action="store_true", default=False, help="list modes?")

parser.add_option("-c", "--crop-husky", dest="crop", action="store_true", default=False, help="crop out the image header from the Husky?")

parser.add_option("-p", "--playback", dest="playback", action="store_true", default=False, help="just play back the frames, don't process them")

parser.add_option("--save", dest="save", action="store", default=None, help="directory to save the rendered frames to")

parser.add_option("--headless", dest="headless", action="store_true", default=False, help="processing only: show no vizualizations")

(options, args) = parser.parse_args()




if not options.headless:
    cv2.namedWindow("display", cv2.cv.CV_WINDOW_NORMAL)
mog = cv2.BackgroundSubtractorMOG()




def original(img):
    return img

def grayscale(img):
    return cv2.cvtColor(img, cv.CV_BGR2GRAY)

def sobelx(img):
    img_grey = cv2.cvtColor(img, cv.CV_BGR2GRAY)
    gradient = cv2.Sobel(img_grey, ddepth=cv.CV_64F, dx=1, dy=0, ksize=5)
    return np.uint8(np.absolute(gradient))

def sobely(img):
    img_grey = cv2.cvtColor(img, cv.CV_BGR2GRAY)
    gradient = cv2.Sobel(img_grey, ddepth=cv.CV_64F, dx=0, dy=1, ksize=5)
    return np.uint8(np.absolute(gradient))

def sobelboth(img):
    img_grey = cv2.cvtColor(img, cv.CV_BGR2GRAY)
    gradient = cv2.Sobel(img_grey, ddepth=cv.CV_64F, dx=1, dy=1, ksize=5)
    return np.uint8(np.absolute(gradient))

def laplacian(img):
    img_grey = cv2.cvtColor(img, cv.CV_BGR2GRAY)
    lapl = cv2.Laplacian(img_grey, cv2.CV_64F)
    return cv2.convertScaleAbs(lapl)

def canny(img):
    img_grey = cv2.cvtColor(img, cv.CV_BGR2GRAY)
    return cv2.Canny(img_grey, 100, 200)

def gaussian(img):
    return cv2.GaussianBlur(img, (5,5), 1)

def bilateral(img):
    return cv2.bilateralFilter(img, 9, 75, 75)

def segmented(img):
    (segmented_image, labels_image, number_regions) = pms.segment(img, spatial_radius=6, range_radius=4.5, min_density=50)
    return segmented_image

def segmented_downsampled(img):
    downsampled = cv2.resize(img, (0,0), fx=0.25, fy=0.25)
    (segmented_image, labels_image, number_regions) = pms.segment(downsampled, spatial_radius=6, range_radius=4.5, min_density=50)
    return segmented_image

def scharrx(img):
    img_grey = cv2.cvtColor(img, cv.CV_BGR2GRAY)
    return cv2.Scharr(img_grey, ddepth=cv.CV_64F, dx=1, dy=0)

def scharry(img):
    img_grey = cv2.cvtColor(img, cv.CV_BGR2GRAY)
    return cv2.Scharr(img_grey, ddepth=cv.CV_64F, dx=0, dy=1)

def scharrboth(img):
    img_grey = cv2.cvtColor(img, cv.CV_BGR2GRAY)
    return cv2.Scharr(img_grey, ddepth=cv.CV_64F, dx=1, dy=1)

def opencv_segmentation(img):
    return cv2.pyrMeanShiftFiltering(img, sp=12, sr=9)

def grabcut(img):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    fg = np.zeros((1,65), np.float64)
    bg = np.zeros((1,65), np.float64)
    return cv2.grabCut(img, mask, None, bg, fg, 10)

def hough_circles(img):
    img_grey = cv2.cvtColor(img, cv.CV_BGR2GRAY)
    circles = cv2.HoughCircles(img_grey, cv2.cv.CV_HOUGH_GRADIENT, 1, 10, param1=100, param2=30, minRadius=5, maxRadius=20)
    if circles == None:
        circles = []
    for circle_list in circles:
        for circle in circle_list:
            cv2.circle(img, (int(circle[0]), int(circle[1])), int(circle[2]), (255,0,0))
    return img

def hough_lines(img):
    edge = cv2.cvtColor(img, cv.CV_BGR2GRAY)
    lines = cv2.HoughLines(edge, 1, np.pi/180, 130)
    edge = cv2.cvtColor(edge, cv.CV_GRAY2RGB)
    if lines == None:
        lines = []
    for line_list in lines:
        for rho,theta in line_list:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))   # Here i have used int() instead of rounding the decimal value, so 3.8 --> 3
            y1 = int(y0 + 1000*(a))    # But if you want to round the number, then use np.around() function, then 3.8 --> 4.0
            x2 = int(x0 - 1000*(-b))   # But we need integers, so use int() function after that, ie int(np.around(x))
            y2 = int(y0 - 1000*(a))
            cv2.line(edge,(x1,y1),(x2,y2),(255,0,0))
    return edge

def hough_circles_edge(img):
    img_grey = cv2.cvtColor(img, cv.CV_BGR2GRAY)
    img_grey = cv2.Canny(img_grey, 100, 200)
    circles = cv2.HoughCircles(img_grey, cv2.cv.CV_HOUGH_GRADIENT, 1, 10, param1=100, param2=30, minRadius=5, maxRadius=20)
    if circles == None:
        circles = []
    for circle_list in circles:
        for circle in circle_list:
            cv2.circle(img, (int(circle[0]), int(circle[1])), int(circle[2]), (255,0,0))
    return img

def hough_lines_edge(img):
    img_grey = cv2.cvtColor(img, cv.CV_BGR2GRAY)
    edge = cv2.Canny(img_grey, 100, 200)
    lines = cv2.HoughLines(edge, 1, np.pi/180, 130)
    edge = cv2.cvtColor(edge, cv.CV_GRAY2RGB)
    if lines == None:
        lines = []
    for line_list in lines:
        for rho,theta in line_list:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))   # Here i have used int() instead of rounding the decimal value, so 3.8 --> 3
            y1 = int(y0 + 1000*(a))    # But if you want to round the number, then use np.around() function, then 3.8 --> 4.0
            x2 = int(x0 - 1000*(-b))   # But we need integers, so use int() function after that, ie int(np.around(x))
            y2 = int(y0 - 1000*(a))
            cv2.line(edge,(x1,y1),(x2,y2),(255,0,0))
    return edge

def harris_corners(img):
    img_grey = cv2.cvtColor(img, cv.CV_BGR2GRAY)
    dst = cv2.cornerHarris(img_grey, 2, 3, 0.04)
    dst = cv2.dilate(dst,None)
    img[dst>0.01*dst.max()]=[0,0,255]
    return img

def harris_corners_edge(img):
    img_grey = cv2.cvtColor(img, cv.CV_BGR2GRAY)
    img_grey = cv2.Canny(img_grey, 100, 200)
    dst = cv2.cornerHarris(img_grey, 2, 3, 0.04)
    dst = cv2.dilate(dst,None)
    img[dst>0.01*dst.max()]=[0,0,255]
    cv2.imshow("display", img)

def background_subtraction(img):
    return mog.apply(img)

def histogram_equalization(img):
    img_grey = cv2.cvtColor(img, cv.CV_BGR2GRAY)
    return cv2.equalizeHist(img_grey)

def contours(img):
    img_grey = cv2.cvtColor(img, cv.CV_BGR2GRAY)
    edge = cv2.Canny(img_grey, 100, 200)
    (contours, hierarchy) = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        color = np.random.randint(0,255,(3)).tolist()
        cv2.drawContours(img, [cnt], 0, color, 2)
    return img
 
def moments(img):
    img_grey = cv2.cvtColor(img, cv.CV_BGR2GRAY)
    edge = cv2.Canny(img_grey, 100, 200)
    (contours, hierarchy) = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        moments = cv2.moments(cnt)
        if moments['m00'] != 0:
            cx = int(moments['m10']/moments['m00'])
            cy = int(moments['m01']/moments['m00'])
            moment_area = moments['m00']
            contour_area = cv2.contourArea(cnt)

            cv2.drawContours(img, [cnt], 0, (0,255,0), 1)
            cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
    return img


options.mode = "m%s" % options.mode

modes = dict(
        m0=("original", original),
        m1=("grayscale", grayscale),
        m2=("sobel gradient x", sobelx),
        m3=("sobel gradient y", sobely),
        m4=("sobel gradient x and y", sobelboth),
        m5=("laplacian", laplacian),
        m6=("canny edges", canny),
        m7=("gaussian blur", gaussian),
        m8=("bilateral blur/filter", bilateral),
        m9=("mean shift segmentation", segmented),
        m10=("downsampled segmentation", segmented_downsampled),
        m11=("scharr gradient x", scharrx),
        m12=("scharr gradient y", scharry),
        m13=("opencv mean shift (segmentation?)", opencv_segmentation),
        m14=("grabcut (segmentation?)", grabcut),
        m15=("hough circle detection", hough_circles),
        m16=("hough line detection", hough_lines),
        m17=("hough circles (edge image)", hough_circles_edge),
        m18=("hough lines (edge image)", hough_lines_edge),
        m19=("harris corner detection", harris_corners),
        m20=("harris corners (edge image)", harris_corners_edge),
        m21=("background subtraction (MOG)", background_subtraction),
        m22=("histogram equalization", histogram_equalization),
        m23=("contours", contours),
        m24=("moments", moments)
        )


if options.list:
    print "list of modes:"
    keys = modes.keys()
    keys.sort()
    for key in keys:
        print "%s: %s" % (key[1:], modes[key][0])
    sys.exit(0)


framerate = int(options.framerate)
frame = int(options.start_frame)
try:
    while True:
        framefile = "%s%sframe%04d.jpg" % (options.input_dir, os.sep, frame)
        print framefile

        if not os.path.isfile(framefile):
            break

        img = cv2.imread(framefile)
        if options.crop:
            img = img[20:, :]

        start_time = time.time()
    
        if options.playback:
            if not options.headless:
                cv2.imshow("display", img)
        else:
            img = modes[options.mode][1](img)
            if not options.headless:
                cv2.imshow("display", img)
            
        if options.save != None:
            outfile = "%s%sframe%04d.jpg" % (options.save, os.sep, frame)
            cv2.imwrite(outfile, img)

        end_time = time.time()
        print "Frame took %s seconds to process" % (end_time-start_time)

        cv2.waitKey(int(1.0/framerate*1000))
        frame += 1

except IOError, e:
    pass # DONE
