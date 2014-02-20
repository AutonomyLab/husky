#! /usr/bin/env python

from __future__ import print_function
import sys, cv, cv2, os
import numpy as np
import subprocess, signal
import math
import atexit
import cPickle as pickle
from sklearn.cluster import DBSCAN
from sklearn import metrics, preprocessing

#-----------------------------------------------------------------

class MotionDetector:
    
    def __init__(self, config):
        self.__dict__.update(config)

        self.fgbg = cv2.BackgroundSubtractorMOG()
        self.hysteresis_sum = None

        # size of the image. updated each frame
        self.WIDTH = 0
        self.HEIGHT = 0

        # color depth, probably 3
        self.DEPTH = 3

        # motion detections
        self.detections = set()

        # publish every time this equals publish interval
        self.frame_count = 0

#-----------------------------------------------------------------

    def reset_detector(self):
        self.detections = set()
        # reset our background subtractor periodically
        self.fgbg = cv2.BackgroundSubtractorMOG()
        self.frame_count = 0

#-----------------------------------------------------------------

    def apply_frame(self, img):
        self.image = img

        self.HEIGHT, self.WIDTH, self.DEPTH = self.image.shape

        # BACKGROUND SUBTRACTION
        fgmask = self.fgbg.apply(self.image)
        
        # FILTERING FOR MOTION AREAS
        significant_motion = self.find_significant_motion_from_mask(fgmask)

        # HYSTERESIS
        rectangles = self.apply_hysteresis(significant_motion)

        self.detections.update(rectangles)

        self.frame_count += 1
        
        self.VIZ_CALLBACK(self.detections, self.image)

        if self.frame_count >= self.publish_interval:
            # PUBLISH SUPER-POLYGON
            self.CALLBACK(self.detections)
            self.reset_detector()

#-----------------------------------------------------------------
 
    def apply_hysteresis(self, significant_motion):
        if self.hysteresis_sum == None:
            self.hysteresis_sum = np.zeros((self.WIDTH/self.MOTION_WINDOW_SIZE, self.HEIGHT/self.MOTION_WINDOW_SIZE))

        hysteresis = np.zeros((self.WIDTH/self.MOTION_WINDOW_SIZE, self.HEIGHT/self.MOTION_WINDOW_SIZE))

        for rect in significant_motion:
            coord = (rect[0][0] / self.MOTION_WINDOW_SIZE, rect[0][1] / self.MOTION_WINDOW_SIZE)
            hysteresis[coord] = 1

        rectangles = []
        for ri,row in enumerate(self.hysteresis_sum):
            for ci,cell in enumerate(row):
                if cell >= self.MIN_HYSTERESIS_FRAMES - 1:
                    coord = (ri,ci)
                    # top left
                    p0 = (coord[0] * self.MOTION_WINDOW_SIZE, coord[1] * self.MOTION_WINDOW_SIZE)

                    # top right
                    p1 = (p0[0] + self.MOTION_WINDOW_SIZE-1, p0[1])
                    
                    # bottom right
                    p2 = (p0[0] + self.MOTION_WINDOW_SIZE-1, p0[1] + self.MOTION_WINDOW_SIZE-1)
                    
                    # bottom left
                    p3 = (p0[0], p0[1] + self.MOTION_WINDOW_SIZE-1)

                    rectangles.append((p0, p1, p2, p3))


        # increment persistent frame counts
        self.hysteresis_sum += hysteresis

        # decrement the non-persistent counts
        decrement = hysteresis
        decrement -= np.ones_like(decrement)
        decrement *= (np.zeros_like(decrement) - np.ones_like(decrement))
        # now decrement[x,y] == 1 iff hysteresis[x,y] == 0
        self.hysteresis_sum -= decrement * self.DECAY_RATE
        self.hysteresis_sum = np.clip(self.hysteresis_sum, 0, 3*self.MIN_HYSTERESIS_FRAMES)


        return rectangles

#-----------------------------------------------------------------

    def find_significant_motion_from_mask(self, mask):
        sig = []

        size = self.MOTION_WINDOW_SIZE
        threshold = self.MOTION_THRESHOLD

        x = 0
        y = 0

        while x+size < self.WIDTH:
            y = 0
            while y+size < self.HEIGHT:
                subimage = mask[y:y+size, x:x+size]
                if np.sum(subimage)/255 >= threshold:
                    sig.append(((x,y),(x+size-1,y+size-1)))
                y += size
            x += size

        return sig

#-----------------------------------------------------------------
# END CLASS MOTIONDETECTOR
#-----------------------------------------------------------------

#-----------------------------------------------------------------

class OscillationDetector:
    
    def __init__(self, config):
        self.__dict__.update(config)

        # calculate the spectral bins using
        # the desired frequency range, the camera framerate
        # and the size of the temporal window
        bin_width = self.CAMERA_FRAMERATE / float(self.TEMPORAL_WINDOW)
        smallest_bin = int(self.MIN_GESTURE_FREQUENCY / bin_width) + 1
        freq_range = self.MAX_GESTURE_FREQUENCY - self.MIN_GESTURE_FREQUENCY
        largest_bin = smallest_bin + int(freq_range / bin_width)
        if smallest_bin < 2:
            smallest_bin += 1
            largest_bin += 1

        # these are our frequency bins of interest
        self.FREQ_COMPONENTS = range(smallest_bin, largest_bin+1)

        # averaging kernel.
        # we use a Gaussian centered in the middle of the window
        # with a variance equal to twice the size of the window
        self.kernel = np.zeros((self.SPATIAL_WINDOW_X, self.SPATIAL_WINDOW_Y))
        varx = self.SPATIAL_WINDOW_X * 2.0
        vary = self.SPATIAL_WINDOW_Y * 2.0

        from scipy import signal
        xwindow = signal.gaussian(self.SPATIAL_WINDOW_X, std=math.sqrt(varx))
        ywindow = signal.gaussian(self.SPATIAL_WINDOW_Y, std=math.sqrt(vary))

        for x in range(0, self.SPATIAL_WINDOW_X):
            for y in range(0, self.SPATIAL_WINDOW_Y):
                self.kernel[x,y] = xwindow[x] * ywindow[y]

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

        # clusters and individual regions
        self.clusters = []
        self.outliers = []
        self.periodic_windows = []

#-----------------------------------------------------------------

    def set_motion_areas(self, areas):
        self.motion_areas = areas

#-----------------------------------------------------------------

    def reset_detector(self):
        self.temporal_window_index = 0
        self.temporal_window_full = False

#-----------------------------------------------------------------

    def apply_frame(self, img):
        self.image = img

        # wait for motion detection first
        if self.motion_areas == None:
            return

        self.HEIGHT, self.WIDTH = self.image.shape

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

        # visualization
        self.VIZ_CALLBACK(self.periodic_windows, self.clusters, self.outliers)

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
            bottom_right = area[2]

            x = top_left[0]
            y = top_left[1]

            while x <= bottom_right[0] and y <= bottom_right[1]:
                if flagged_areas[x,y] == 0:
                    # add rectangle to regions of interest

                    window = ((x,y,
                        x+self.SPATIAL_WINDOW_X,
                        y+self.SPATIAL_WINDOW_Y),
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

        if not self.temporal_window_full:
            return 

        self.periodic_windows = []
        X = []

        for window in self.spatial_windows:
            time_domain = np.asarray(window[1])
            avg_intensity = np.mean(time_domain)
            frequency_domain = abs(np.fft.fft(time_domain))

            # use mean?
            # avg = np.sum(frequency_domain[1:len(frequency_domain)/2]) / (float(len(frequency_domain)/2-1))
            
            # use median?
            avg = np.median(frequency_domain[1:len(frequency_domain)/2])

            std = np.std(frequency_domain[1:len(frequency_domain)/2])

            threshold = max(avg + std*self.PEAK_STDEVS, self.MIN_FREQ_INTENSITY)

            periodic = False
            avg_bin = 0
            total_energy = 1
            highest = 0
            for f in self.FREQ_COMPONENTS:
                avg_bin += f*frequency_domain[f]
                total_energy += frequency_domain[f]
                if frequency_domain[f] > highest:
                    highest = frequency_domain[f]
                if (frequency_domain[f] >= threshold and
                        frequency_domain[f] > frequency_domain[1]):
                    periodic = True

            avg_bin /= float(total_energy)

            if periodic:
                self.periodic_windows.append(window)

                # features for clustering
                X.append((float(window[0][0] + self.SPATIAL_WINDOW_X/2),
                        float(window[0][1] + self.SPATIAL_WINDOW_Y/2),
                        avg_bin # frequency seems like an important clue
                        #avg_intensity
                        ))

        self.clusters = []
        self.outliers = []
        if len(X) > 0:
            # cluster these positive rectangles somehow so we can
            # have multiple different periodic motions detected in the
            # same scene
            X = np.asarray(X)

            # normalize features
            scaled_X = preprocessing.scale(X)

            # cluster with DBSCAN
            db = DBSCAN(eps=self.EPS, min_samples=self.MIN_SAMPLES).fit(scaled_X)

            # process results
            core_samples = db.core_sample_indices_
            labels = db.labels_

            # Number of clusters in labels, ignoring noise if present.
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)


            unique_labels = set(labels)
            for k in unique_labels:
                class_members = [index[0] for index in np.argwhere(labels == k)]

                min_x = sys.maxint
                max_x = -1
                min_y = sys.maxint
                max_y = -1

                for index in class_members:
                    x = X[index]
                    if k == -1:
                        x0 = x[0]-self.SPATIAL_WINDOW_X/2
                        y0 = x[1]-self.SPATIAL_WINDOW_Y/2
                        x1 = x0+self.SPATIAL_WINDOW_X
                        y1 = y0+self.SPATIAL_WINDOW_Y
                        self.outliers.append(((x0,y0,x1,y1), [], True))
                    if x[0] < min_x:
                        min_x = x[0]
                    if x[0] > max_x:
                        max_x = x[0]
                    if x[1] < min_y:
                        min_y = x[1]
                    if x[1] > max_y:
                        max_y = x[1]

                height = max_y - min_y
                width = max_x - min_x

                height = max(width,height)
                height = width = height + self.SPATIAL_WINDOW_X*2

                center_x = (max_x + min_x) / 2
                center_y = (max_y + min_y) / 2

                top_left = (int(center_x - width/2), 
                        int(center_y - height/2))
                bottom_right = (int(center_x + width/2), 
                        int(center_y + height/2))

                # don't consider outliers a cluster
                if k != -1:
                    self.clusters.append((top_left, bottom_right))

        self.DETECTION_CALLBACK(self.clusters)

#-----------------------------------------------------------------

    def average_each_subwindow(self):
        # get an average of the pixels in each window
        # of interest that we can use at the end of the round,
        # or at each frame of the round once the temporal window
        # is full

        area = float(self.SPATIAL_WINDOW_X*self.SPATIAL_WINDOW_Y)

        for window in self.spatial_windows:
            ymin = window[0][1]
            ymax = window[0][3]

            xmin = window[0][0]
            xmax = window[0][2]

            subimage = self.image[ymin:ymax, xmin:xmax]

            # apply averaging kernel
            total = np.sum(np.multiply(subimage, self.kernel))

            avg = total / area

            window[1].append(avg)

#-----------------------------------------------------------------
# END CLASS OSCILLATION DETECTOR
#-----------------------------------------------------------------



# TODO: tune the detector to maximize true positive rate



from optparse import OptionParser

parser = OptionParser()

parser.add_option("-i", "--input", dest="input_dir",
        help="directory with annotated frames")

parser.add_option("-a", "--annotations", dest="ann_dir",
        help="directory to find annotations")

parser.add_option("--min_peak", dest="min_peak", action="store", type="int",
        help="minimum size of a frequency bin")

parser.add_option("--peak_sens", dest="peak_sens", action="store", type="float",
        help="how many stdevs for the peak to be an outlier")

parser.add_option("--cluster_eps", dest="cluster_eps", action="store", type="float",
        help="maximum distance linking neighbours in clustering alg")

parser.add_option("--cluster_min_samples", dest="cluster_min_samples", action="store", type="int",
        help="minimum number of neighbours to be considered a core sample in clustering alg")

parser.add_option("-v", "--viz", dest="viz", action="store_true", default=False, help="show viz window?")

parser.add_option("-m", "--motion", dest="motion", action="store_true", default=False, help="show motion viz window?")

parser.add_option("-e", "--edge", dest="edge", action="store_true", default=False, help="use edge images as input?")

parser.add_option("--sobel", dest="sobel", action="store_true", default=False, help="apply sobel gradient operator to image?")

parser.add_option("-b", "--blur", dest="blur", action="store_true", default=False, help="blur input image first?")

parser.add_option("-l", "--laplacian", dest="laplacian", action="store_true", default=False, help="apply laplacian operator to image?")

parser.add_option("-s", "--start", dest="start", action="store", default=0, help="start on this frame")

(options, args) = parser.parse_args()
options.start = int(options.start)

motion_areas = []
periodic_clusters = []

motion = None
gestures = None

def noop(arg1=0, arg2=0, arg3=0):
    pass

if options.viz:
    cv2.namedWindow("viz", cv2.cv.CV_WINDOW_NORMAL)

if options.motion:
    cv2.namedWindow("motion", cv2.cv.CV_WINDOW_NORMAL)

display_img = None
def viz_callback(periodic_regions, clusters, outliers):
    if options.viz:
        img = display_img

        for rect in periodic_regions:
            p0 = (int(rect[0][0]), int(rect[0][1]))
            p1 = (int(rect[0][2]), int(rect[0][3]))
            cv2.rectangle(img, p0, p1, (255, 255, 255))

        for rect in clusters:
            p0 = rect[0]
            p1 = rect[1]
            cv2.rectangle(img, p0, p1, (0, 255, 0))

        for rect in outliers:
            p0 = (int(rect[0][0]), int(rect[0][1]))
            p1 = (int(rect[0][2]), int(rect[0][3]))
            cv2.rectangle(img, p0, p1, (0, 0, 100))

        cv2.imshow("viz", img)
        cv2.waitKey(1)

def motion_viz_callback(motion, img):
    if options.motion:
        for rect in motion:
            cv2.rectangle(img, rect[0], rect[2], (255, 255, 255))

        cv2.imshow("motion", img)
        cv2.waitKey(1)
 

def motion_callback(detections):
    global motion_areas, gestures
    motion_areas = list(detections)
    gestures.set_motion_areas(motion_areas)

def oscillation_callback(clusters):
    global periodic_clusters
    periodic_clusters = clusters

motion = MotionDetector(dict(
    MOTION_WINDOW_SIZE = 25,
    MOTION_THRESHOLD = 1,
    MIN_HYSTERESIS_FRAMES = 1,
    DECAY_RATE = 1,
    publish_interval = 30,
    CALLBACK = motion_callback,
    VIZ_CALLBACK = motion_viz_callback))

gestures = OscillationDetector(dict(
    TEMPORAL_WINDOW = 60,
    MIN_GESTURE_FREQUENCY = 0.5,
    MAX_GESTURE_FREQUENCY =  3,
    CAMERA_FRAMERATE = 30,
    PEAK_STDEVS = options.peak_sens,
    MIN_FREQ_INTENSITY = options.min_peak,
    SPATIAL_WINDOW_X = 10,
    SPATIAL_WINDOW_Y = 10,
    OVERLAP_FACTOR = 2,
    EPS = options.cluster_eps,
    MIN_SAMPLES = options.cluster_min_samples,
    DETECTION_CALLBACK = oscillation_callback,
    VIZ_CALLBACK = viz_callback))

def dist_squared(cluster, annotation):
    center = ((cluster[0][0]+cluster[1][0])/2, (cluster[0][1]+cluster[1][1])/2)
    return (center[0]-annotation[0])**2 + (center[1]-annotation[1])**2

total_error = 0
total_clusters = 0

input_dirs = options.input_dir.split(",")
ann_dirs = options.ann_dir.split(",")

for i in range(0,len(input_dirs)):
    input_dir = input_dirs[i]
    ann_dir = ann_dirs[i]

    start = options.start
    frame = start

    picfile = "%s%sannotations.pic" % (ann_dir, os.sep)
    all_annotations = []
    with open(picfile) as f:
        all_annotations = pickle.load(f)

    while True:
        framefile = "%s%sframe%04d.jpg" % (input_dir, os.sep, frame)

        if not os.path.isfile(framefile):
            break

        img = cv2.imread(framefile)
        img_grey = cv2.imread(framefile, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        display_img = img

        # use edge images as input
        if options.edge:
            img_grey = cv2.Canny(img_grey, 100, 200)
            display_img = cv2.cvtColor(img_grey, cv2.COLOR_GRAY2BGR)

        if options.laplacian:
            img_grey = cv2.Laplacian(img_grey, cv2.CV_64F)
            img_grey = cv2.convertScaleAbs(img_grey)
            display_img = cv2.cvtColor(img_grey, cv2.COLOR_GRAY2BGR)

        if options.sobel:
            img_grey = cv2.Sobel(img_grey, ddepth=cv.CV_64F, dx=1, dy=1, ksize=5)
            img_grey = np.uint8(np.absolute(img_grey))
            display_img = cv2.cvtColor(img_grey, cv2.COLOR_GRAY2BGR)

        if options.blur:
            img_grey = cv2.GaussianBlur(img_grey, (5,5), 1)
            display_img = cv2.cvtColor(img_grey, cv2.COLOR_GRAY2BGR)
        
        annotations = all_annotations[frame]
        
        motion.apply_frame(img)
        gestures.apply_frame(img_grey)

        total_clusters += len(periodic_clusters)

        for cluster in periodic_clusters:
            if len(annotations) > 0:
                closest_dist = sys.maxint
                closest_a = None
                for a in annotations:
                    dist = dist_squared(cluster, a)
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_a = a

                total_error += min(1.0 / (closest_dist+0.01), 0.1)
            else:
                total_error -= 0.001

        if len(periodic_clusters) == 0:
            total_error -= 0.001 * len(annotations)

        frame += 1

msg = "sens=%s, min_peak=%s, eps=%s, min_samples=%s, score=%s, total_clusters=%s" % (options.peak_sens, options.min_peak, options.cluster_eps, options.cluster_min_samples, total_error, total_clusters)
print(msg)
