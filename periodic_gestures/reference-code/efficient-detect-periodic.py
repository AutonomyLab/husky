#! /usr/bin/env python

# This script uses Fourier analysis to try to detect regions of the image
# with periodicity in a certain frequency range.

import cv, cv2, time
import numpy as np, matplotlib.pyplot as plot
import sys, os, cPickle as pickle


TEMPORAL_WINDOW = 120 # number of frames to consider in Fourier analysis
FREQ_COMPONENTS = [3,4,5,6,7]

TEMPORAL_WINDOW_FULL = False

# SPIKE_DETECTION = "factor"
# SPIKE_DETECTION = "offset"
# SPIKE_DETECTION = "combo"
# SPIKE_DETECTION = "max" # if the spike is maximal
SPIKE_DETECTION = "outlier" # if the spike is a statistical outlier

# standard deviations above the mean required to be considered a spike
PEAK_STDEVS = 9

# frequency bin must be above this in order to be considered a significant peak
MIN_FREQ_INTENSITY = 50

# size of overlapping bounding boxes to check for periodicity
SPATIAL_WINDOW_X = 10
SPATIAL_WINDOW_Y = 10

# overlap factor of 2 means each window overlaps by half
OVERLAP_FACTOR = 2

# scale to resize optical flow windows to, for consistent SVM inputs
TARGET_SCALE_X = 50
TARGET_SCALE_Y = 50

# color of the rectangles to draw over the image
OVERLAY_COLOR = (255, 255, 255)

# the box to monitor for freq histogram or waveform
# MONITOR_BOX = 310 # 25 meter, foot area when walking
# MONITOR_BOX = 239 # 25 meter, periodic box
# MONITOR_BOX = 235 # 25 meter, to the left of person
MONITOR_BOX = 0
MONITOR_BOX_COLOR = (255, 0, 0)
MONITOR_BOX_COLOR_POSITIVE = (0,255,0)

# if we want to see the frequency histogram in the desired box
DISPLAY_HIST = False
FFT = [] # the FFT of the monitor box
FFT_THRESHOLD = 0 # this is set per frame

# if we want to see the FFT of the FFT, for tuning purposes.
DISPLAY_SECOND_DEGREE_FOURIER = False
SECOND_DEGREE_FFT = []

# if we want to see the intensity waveform in the desired box
DISPLAY_WAVEFORM = False
WAVEFORM = [] # the waveform of the monitor box

# display a window with the optical flow?
OPTICAL_FLOW = False
OPTICAL_FLOW_IMAGE = None

# display a window with the cropped rectangle's optical flow?
CROPPED_OPTICAL_FLOW = False
CROPPED_OPTICAL_FLOW_IMAGE = None

# draw a color representation of the flow instead of a vector field?
OPTICAL_FLOW_HSV = False 

# display the image fullscreen?
FULLSCREEN = False

# if we just want to scan through the images
SCAN = False

# show the mask of the background subtraction
MASK = False

# set once we load an image
WIDTH = 0
HEIGHT = 0

START_FRAME = 0
DISPLAY_IMAGE = None

# vector of windows, each of which is a time-indexed vector of avg intensities
windows = []

# these are where we'll draw our bounding boxes
motion_detected_windows = []

# we'll try and create some bounding boxes that contain all the periodic motion
PERIODIC_RECTANGLES = []
PERIODIC_RECTANGLE_COLOR = (0,0,255)

# for visualization purposes
CURRENT_PERIODIC_RECTANGLES = []

# we'll use this to label the regions we're finding, for output
PERIODIC_RECTANGLE_INDEX = 0

# export 30 frames of each candidate region to the classifier
CLASSIFIER_REGION_PERIOD = 30

# directory to save cropped regions to
OUTDIR = None

# hide all windows, for batch processing
HEADLESS = False

# don't save images, just optical flows
SKIP_IMAGES = False

# motion detection parameters
MOTION_WINDOW_SIZE = 25
MOTION_THRESHOLD = 1

# how long to require motion to stick around
MIN_HYSTERESIS_FRAMES = 3

# show edge image
EDGE_IMAGE = False

# how quickly to decay non-persistent cells
DECAY_RATE = 2

# playback framerate
FRAMERATE = 0.03

# the last frame, for optical flow or whatever
last_frame = None
last_color_frame = None

#-----------------------------------------------------------------


def initialize():
    global windows, WIDTH, HEIGHT

    imagefile = "frame%04d.jpg" % 1
    path = "%s%s" % (directory, imagefile)
    image = cv.LoadImage(path, cv.CV_LOAD_IMAGE_GRAYSCALE)
    WIDTH, HEIGHT = cv.GetSize(image)

    num_windows_x = WIDTH / (SPATIAL_WINDOW_X / OVERLAP_FACTOR)
    num_windows_y = HEIGHT / (SPATIAL_WINDOW_Y / OVERLAP_FACTOR)
    num_windows = num_windows_x * num_windows_y
    for k in range(0, num_windows):
        temporal_window = [0] * TEMPORAL_WINDOW
        windows.append(temporal_window)

    if not HEADLESS:
        if DISPLAY_HIST:
            plot.figure(1)
            plot.show(block=False)

        if DISPLAY_WAVEFORM:
            plot.figure(2)
            plot.show(block=False)

        if DISPLAY_SECOND_DEGREE_FOURIER:
            plot.figure(3)
            plot.show(block=False)

        cv.NamedWindow("display", cv.CV_WINDOW_NORMAL)
        if FULLSCREEN:
            cv2.setWindowProperty("display", cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)

        cv.SetMouseCallback("display", handle_mouse)

        if OPTICAL_FLOW:
            cv.NamedWindow("optical_flow", cv.CV_WINDOW_NORMAL)
        
        if CROPPED_OPTICAL_FLOW:
            cv.NamedWindow("cropped_optical_flow", cv.CV_WINDOW_NORMAL)




#-----------------------------------------------------------------


def detect_periodic(directory):
    global windows, WIDTH, HEIGHT, WAVEFORM, FFT_THRESHOLD, last_frame, last_color_frame, motion_detected_windows, TEMPORAL_WINDOW_FULL, PERIODIC_RECTANGLES, DISPLAY_IMAGE, CURRENT_PERIODIC_RECTANGLES, OPTICAL_FLOW_IMAGE

    initialize()
    feature_params = dict( maxCorners = 100,
                            qualityLevel = 0.003,
                            minDistance = 7,
                            blockSize = 7 )
    lk_params = dict( winSize = (15,15),
                        maxLevel = 2,
                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03) )

    color = np.random.randint(0,255,(100,3))

    fgbg = cv2.BackgroundSubtractorMOG()

    hysteresis = None

    i = START_FRAME
    # iterate until we can no longer read an image
    while True:
        frame_st = time.time()
        imagefile = "frame%04d.jpg" % (i+1)
        print imagefile
        path = "%s%s" % (directory, imagefile)
        image = cv.LoadImage(path)
        gray_image = cv.LoadImage(path, cv.CV_LOAD_IMAGE_GRAYSCALE)
        color_image = cv.LoadImage(path)
        WIDTH, HEIGHT = cv.GetSize(image)

        if i == START_FRAME:
            hysteresis = np.zeros((WIDTH/MOTION_WINDOW_SIZE, HEIGHT/MOTION_WINDOW_SIZE))
            hysteresis_sum = hysteresis
            last_frame = image	

        # identify regions with sufficient optical flow

        # calculate optical flow
        cv2_last = np.asarray(cv.GetMat(last_frame))
        cv2_next = np.asarray(cv.GetMat(image))
        cv2_gray = np.asarray(cv.GetMat(gray_image))
        cv2_color = np.asarray(cv.GetMat(color_image))
        #start = time.time()

        #flow = cv2.calcOpticalFlowFarneback(cv2_last, cv2_next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        #if (i-START_FRAME+1) % 60 == 1:
        #    p0 = cv2.goodFeaturesToTrack(cv2_last, mask = None, **feature_params)
        #    mask = np.zeros_like(np.asarray(cv.GetMat(image)))

        #mid = time.time()

        #p1, st, err = cv2.calcOpticalFlowPyrLK(cv2_last, cv2_next, p0, None, **lk_params)

        #end = time.time()

        #good_new = p1[st==1]
        #good_old = p0[st==1]

        # draw the tracks
        #for idx,(new,old) in enumerate(zip(good_new,good_old)):
        #    a,b = new.ravel()
        #    c,d = old.ravel()
        #    cv2.line(mask, (a,b),(c,d), color[idx].tolist(), 2)
        #    cv2.circle(cv2_next,(a,b),5,color[idx].tolist(),-1)
        #OPTICAL_FLOW_IMAGE = cv2.add(cv2_next,mask)

        #if (i-START_FRAME+1) % 15 == 1:
        #    flow = cv2.calcOpticalFlowFarneback(cv2_last, cv2_next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        #    st = time.time()
        #    significant_motion = find_significant_motion(flow)
        #    end = time.time()
        #    print "Finding significant motion took %s secs" % (end-st)

        st = time.time()
        fgmask = fgbg.apply(cv2_color)
        end = time.time()
        print "Background subtraction took %s secs" % (end-st)

        st = time.time()
        significant_motion = find_significant_motion_from_mask(fgmask)
        end = time.time()
        print "Finding areas of significant motion took %s secs" % (end-st)
        
        colormask = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
        if MASK:
            OPTICAL_FLOW_IMAGE = colormask
        elif EDGE_IMAGE:
            coloredge = cv2.cvtColor(cv2.Canny(cv2_gray, 100, 200), cv2.COLOR_GRAY2BGR)
            OPTICAL_FLOW_IMAGE = coloredge
        else:
            OPTICAL_FLOW_IMAGE = cv2_color
        
        #OPTICAL_FLOW_IMAGE = draw_flow(cv2_next, flow, step=WIDTH / 25)

        hysteresis = np.zeros((WIDTH/MOTION_WINDOW_SIZE, HEIGHT/MOTION_WINDOW_SIZE))
        for rect in significant_motion:
            coord = (rect[0][0] / MOTION_WINDOW_SIZE, rect[0][1] / MOTION_WINDOW_SIZE)
            hysteresis[coord] = 1

        for ri,row in enumerate(hysteresis_sum):
            for ci,cell in enumerate(row):
                if cell >= MIN_HYSTERESIS_FRAMES - 1:
                    coord = (ri,ci)
                    p0 = (coord[0] * MOTION_WINDOW_SIZE, coord[1] * MOTION_WINDOW_SIZE)
                    p1 = (p0[0] + MOTION_WINDOW_SIZE-1, p0[1] + MOTION_WINDOW_SIZE-1)
                    rect = (p0,p1)
                    cv2.rectangle(OPTICAL_FLOW_IMAGE, rect[0], rect[1], (0,0,255))

        # increment persistent frame counts
        hysteresis_sum += hysteresis

        # decrement the non-persistent counts
        decrement = hysteresis
        decrement -= np.ones_like(decrement)
        decrement *= (np.zeros_like(decrement) - np.ones_like(decrement))
        # now decrement[x,y] == 1 iff hysteresis[x,y] == 0
        for n in range(0,DECAY_RATE):
            hysteresis_sum -= decrement
        hysteresis_sum = np.clip(hysteresis_sum, 0, 3*MIN_HYSTERESIS_FRAMES)

        # reset count for windows that weren't present this frame
        #hysteresis_sum = np.multiply(hysteresis_sum, hysteresis)

        display_visuals(image)

        #p0 = p1

        last_frame = image	
        i += 1

        frame_end = time.time()
        if (frame_end-frame_st) < FRAMERATE:
            time.sleep(FRAMERATE - (frame_end-frame_st))
        else:
            print "Overloaded."

#-----------------------------------------------------------------

def find_significant_motion_from_mask(mask):
    sig = []

    size = MOTION_WINDOW_SIZE
    threshold = MOTION_THRESHOLD

    x = 0
    y = 0

    while x+size < WIDTH:
        y = 0
        while y+size < HEIGHT:
            subimage = mask[y:y+size, x:x+size]
            if np.sum(subimage)/255 >= threshold:
                sig.append(((x,y),(x+size-1,y+size-1)))
            y += size
        x += size

    return sig

#-----------------------------------------------------------------

def find_significant_motion_from_flow(flow):
    sig = []

    size = 2
    threshold = 0.5

    x = 0
    y = 0

    while x+size < WIDTH:
        y = 0
        while y+size < HEIGHT:
            #total = 0.0
            found = False
            for xi in range(0,size):
                for yi in range(0,size):
                    #total += np.linalg.norm(flow[y+yi, x+xi])
                    if np.linalg.norm(flow[y+yi, x+xi]) > threshold:
                        found = True
            if found: #total/(size**2) > threshold:
                sig.append(((x,y),(x+size,y+size)))
            y += size
        x += size

    return sig

#-----------------------------------------------------------------

def draw_flow(img, flow, step=8):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 0, (0, 255, 0), -1)
    return vis

#-----------------------------------------------------------------

def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

#-----------------------------------------------------------------

def handle_mouse(event, x, y, dummy1, dummy2):
    global MONITOR_BOX
    MONITOR_BOX = get_subwindow_index(x, y)

#-----------------------------------------------------------------


def display_visuals(image):
    DISPLAY_IMAGE = image

    if not HEADLESS:
        DISPLAY_IMAGE = draw_overlays(DISPLAY_IMAGE, motion_detected_windows)
        cv.ShowImage("display", DISPLAY_IMAGE)

        if OPTICAL_FLOW and OPTICAL_FLOW_IMAGE != None:
            cv2.imshow("optical_flow", OPTICAL_FLOW_IMAGE)
        if CROPPED_OPTICAL_FLOW and CROPPED_OPTICAL_FLOW_IMAGE != None:
            cv2.imshow("cropped_optical_flow", CROPPED_OPTICAL_FLOW_IMAGE)

        if DISPLAY_HIST:
            plot.figure(1)
            plot.clf()
            plot.title("Frequency Spectrum")
            plot.bar(range(0, len(FFT)), FFT)
            plot.hlines(FFT_THRESHOLD, 0, len(FFT))
            plot.draw()

        if DISPLAY_WAVEFORM:
            plot.figure(2)
            plot.clf()
            plot.title("Intensity Time-Series")
            plot.axis([0, len(WAVEFORM), 0, 255])
            plot.bar(range(0, len(WAVEFORM)), WAVEFORM)
            plot.draw()

        if DISPLAY_SECOND_DEGREE_FOURIER:
            plot.figure(3)
            plot.clf()
            plot.title("Second Degree FFT")
            plot.bar(range(0, len(SECOND_DEGREE_FFT)), SECOND_DEGREE_FFT)
            plot.draw()

        cv.WaitKey(1)



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

def draw_overlays(image, motion_detected_windows):
    # this function draws a rectangle in all areas of the image
    # specified by motion_detected_windows

    monitor_positive = False
    monitor_loc = get_subwindow_location(MONITOR_BOX)
    for window in motion_detected_windows:
        pt1 = window
        pt2 = (window[0] + SPATIAL_WINDOW_X - 1, window[1] + SPATIAL_WINDOW_Y - 1)
        cv.Rectangle(image, pt1, pt2, OVERLAY_COLOR)
        if (window[0] == monitor_loc[0] and window[1] == monitor_loc[1]):
            monitor_positive = True

    # draw the bounding rectangles for the periodic motion
    for rect in CURRENT_PERIODIC_RECTANGLES:
        cv.Rectangle(image, rect[0], rect[1], PERIODIC_RECTANGLE_COLOR)

    # draw monitor box if we want it
    if DISPLAY_HIST or DISPLAY_WAVEFORM or DISPLAY_SECOND_DEGREE_FOURIER:
        pt1 = get_subwindow_location(MONITOR_BOX, WIDTH, HEIGHT)
        pt2 = (pt1[0] + SPATIAL_WINDOW_X - 1, pt1[1] + SPATIAL_WINDOW_Y - 1)
        if monitor_positive:
            cv.Rectangle(image, pt1, pt2, MONITOR_BOX_COLOR_POSITIVE)
        else:
            cv.Rectangle(image, pt1, pt2, MONITOR_BOX_COLOR)

    return image



#-----------------------------------------------------------------

def main():
    try:
        detect_periodic(directory)
    except IOError:
        print "Either no image was found, or iteration over images complete."
        print "Restarting"
        main()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "Usage: %s <image-directory>" % sys.argv[0]
        sys.exit(1)

    idx = 2
    for arg in sys.argv[2:]:
        if arg == "--hist":
            DISPLAY_HIST = True
        elif arg == "--doublefourier":
            DISPLAY_SECOND_DEGREE_FOURIER = True
        elif arg == "--waveform":
            DISPLAY_WAVEFORM = True
        elif arg == "--fs" or arg == "--fullscreen":
            FULLSCREEN = True
        elif arg == "--outdir":
            OUTDIR = os.path.normpath(sys.argv[idx + 1]) + os.sep
            if not os.path.exists(OUTDIR):
                os.makedirs(OUTDIR)
        elif arg == "--skip-images":
            SKIP_IMAGES = True
        elif arg == "--optical-flow":
            OPTICAL_FLOW = True
        elif arg == "--cropped-optical-flow":
            CROPPED_OPTICAL_FLOW = True
        elif arg == "--hsv":
            OPTICAL_FLOW_HSV = True
        elif arg == "--scan":
            SCAN = True
            DISPLAY_HIST = False
            DISPLAY_WAVEFORM = False
            OUTDIR = None
        elif arg == "--headless":
            HEADLESS = True
        elif arg == "--start":
            START_FRAME = int(sys.argv[idx + 1])
        elif arg == "--mask":
            MASK = True
        elif arg == "--motion-window":
            MOTION_WINDOW_SIZE = int(sys.argv[idx+1])
        elif arg == "--motion-threshold":
            MOTION_THRESHOLD = int(sys.argv[idx+1])
        elif arg == "--framerate":
            FRAMERATE = 1.0 / float(int(sys.argv[idx+1]))
        elif arg == "--hysteresis":
            MIN_HYSTERESIS_FRAMES = int(sys.argv[idx+1])
        elif arg == "--decay-rate":
            DECAY_RATE = int(sys.argv[idx+1])
        elif arg == "--edge":
            EDGE_IMAGE = True

        idx += 1


    directory = os.path.normpath(sys.argv[1]) + os.sep
    main()
