#! /usr/bin/env python

# This node uses MOG background subtraction to detect regions
# of motion in the camera stream

import cv, cv2, time
import numpy as np
import sys, os
import rospy, roslib
from geometry_msgs.msg import Polygon, Point32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

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

class motion_detection:

    def __init__(self):
        rospy.init_node("motion_detection")

        config = dict(
            # motion detection parameters
            MOTION_WINDOW_SIZE = rospy.get_param("~motion_region_size", 25),
            MOTION_THRESHOLD = rospy.get_param("~motion_threshold", 1),

            # how long to require motion to stick around
            MIN_HYSTERESIS_FRAMES = rospy.get_param("~hysteresis_delay", 2),

            # how quickly to decay non-persistent cells
            DECAY_RATE = rospy.get_param("~hysteresis_decay", 1),

            # so we only publish every N frames
            publish_interval = rospy.get_param("~publish_interval", 30),
        
            CALLBACK = self.publish_motion,
            VIZ_CALLBACK = self.publish_viz)

        self.detector = MotionDetector(config)

        self.motion_pub = rospy.Publisher(rospy.get_param("~motion_topic", "motion_detection/motion"), Polygon)

        self.viz_pub = rospy.Publisher(rospy.get_param("~viz_topic", "motion_detection/viz"), Image)

        self.image_sub = rospy.Subscriber(rospy.get_param("~image_topic", "axis/image_raw/decompressed"), Image, self.handle_image)


        self.cv_bridge = CvBridge()

#-----------------------------------------------------------------

    def handle_image(self, data):
        if self.no_one_listening():
            self.detector.reset_detector()
            return

        self.image = self.cv_bridge.imgmsg_to_cv(data, desired_encoding="bgr8")
        self.image = np.asarray(cv.GetMat(self.image))

        self.detector.apply_frame(self.image)

#-----------------------------------------------------------------

    def no_one_listening(self):
        return (self.motion_pub.get_num_connections() < 1 and 
                self.viz_pub.get_num_connections() < 1)

#-----------------------------------------------------------------

    def publish_motion(self, motion):
        super_polygon = []
        for rect in motion:
            super_polygon.append(Point32(x=rect[0][0], y=rect[0][1], z=0))
            super_polygon.append(Point32(x=rect[1][0], y=rect[1][1], z=0))
            super_polygon.append(Point32(x=rect[2][0], y=rect[2][1], z=0))
            super_polygon.append(Point32(x=rect[3][0], y=rect[3][1], z=0))

        self.motion_pub.publish(Polygon(super_polygon))

#-----------------------------------------------------------------

    def publish_viz(self, rectangles, img):
        if self.viz_pub.get_num_connections() < 1:
            return

        for rect in rectangles:
            cv2.rectangle(img, rect[0], rect[2], (255, 255, 255))

        img = cv.fromarray(img)
        msg = self.cv_bridge.cv_to_imgmsg(img, encoding="bgr8")
        self.viz_pub.publish(msg)

#-----------------------------------------------------------------

if __name__ == "__main__":
    detector_node = motion_detection()
    rospy.spin()
