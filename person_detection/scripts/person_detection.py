#! /usr/bin/env python

# This ROS node watches the camera and applies the HOG people
# detection algorithm to find pedestrians.

import roslib
roslib.load_manifest("person_detection")
import sys, rospy, cv, cv2
from std_msgs.msg import String
from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import Image, LaserScan, Joy
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import subprocess, signal
import math
import atexit

#----------------------------------------------------

class person_detection:
    def __init__(self):

        config = dict(
            hog_win_stride = (rospy.get_param("~hog_win_stride",8), rospy.get_param("~hog_win_stride",8)),
            hog_padding = (rospy.get_param("~hog_padding",2), rospy.get_param("~hog_padding",2)),
            hog_scale = rospy.get_param("~hog_scale", 1.075),
            camera_resolution_x = 640,
            camera_resolution_y = 480,
            image_encoding = "bgr8",
            debug = False,
            detected_people_topic = rospy.get_param("~detected_people_topic", "person_detection/people"),
            image_topic = rospy.get_param("~image_topic", "axis/image_raw/decompressed"),
            visualization_topic = rospy.get_param("~visualization_topic", "person_detection/viz"))

        # load object properties from config dict
        self.__dict__.update(config)

        self.setup_ros_node()

        self.person_detection_interval = rospy.Duration(0.75)
        self.person_detection_last = rospy.Time.now()
        self.last_detection = rospy.Time.now()
        # needs to be a Duration object
        self.detection_timeout = rospy.Duration(config["detection_timeout"])
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

#----------------------------------------------------

    def setup_ros_node(self):
        rospy.init_node('person_detection')

        rospy.Subscriber(self.image_topic, Image, self.handle_image)
        self.cv_bridge = CvBridge()
        
        self.people_publisher = rospy.Publisher(self.detected_people_topic, Polygon)
        self.viz_publisher = rospy.Publisher(self.visualization_topic, Image)

#----------------------------------------------------

    def handle_image(self, data):
        # don't waste processing time if no one is listening
        if self.people_publisher.get_num_connections() < 1:
            return

        if rospy.Time.now() - self.person_detection_last < self.person_detection_interval:
            return # don't do anything unless enough time has elapsed

        begin_processing = rospy.Time.now()
        self.person_detection_last = rospy.Time.now()
        
        # convert from ros message to openCV image
        cv2_image = self.ros_msg_to_cv2(data)
        
        # try to detect people
        rectangles = self.get_people_rectangles(cv2_image)
        
        # publish super_polygon
        super_polygon = []
        for rect in rectangles:
            super_polygon.append(Point32((rect[0][0], rect[0][1], 0)))
            super_polygon.append(Point32((rect[1][0], rect[1][1], 0)))
            super_polygon.append(Point32((rect[2][0], rect[2][1], 0)))
            super_polygon.append(Point32((rect[3][0], rect[3][1], 0)))

        self.people_publisher.publish(Polygon(super_polygon))

        if self.viz_publisher.get_num_connections() > 0:
            self.publish_viz(rectangles, cv2_image)
        
        elapsed = rospy.Time.now() - begin_processing
        # adjust frame processing rate to match detector rate,
        # plus a small margin
        self.person_detection_interval = rospy.Duration(elapsed.to_sec() + 0.1)

#----------------------------------------------------

    def publish_viz(self, rectangles, img):
        for rect in rectangles:
            cv2.rectangle(img, rect[0], rect[2], (255, 255, 255))

        msg = cv_bridge.cv_to_imgmsg(img)
        self.viz_publisher.publish(msg)

#----------------------------------------------------

    def get_people_rectangles(self, cv2_image):
        people_found, weights = self.hog.detectMultiScale(
                cv2_image, 
                winStride=self.hog_win_stride, 
                padding=self.hog_padding, 
                scale=self.hog_scale)

        rectangles = []
        for person in people_found:
            p0 = (person[0], person[1])
            p1 = (p0[0] + person[2], p0[1])
            p2 = (p0[0] + person[2], p0[1] + person[3])
            p3 = (p0[0], p0[1] + person[3])

            rectangles.append((p0, p1, p2, p3))

        return rectangles

#----------------------------------------------------

    def ros_msg_to_cv2(self, ros_msg):
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv(ros_msg, desired_encoding=self.image_encoding)
        except CvBridgeError, e:
            print e

        return np.asarray(cv_image[:,:])

#----------------------------------------------------
# END CLASS PERSON_DETECTION
#----------------------------------------------------

if __name__ == "__main__":
    pd = person_detection()
