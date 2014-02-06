#! /usr/bin/env python

# This ROS node listens for periodic motion detection and
# HOG person detection and controls the behavior of the robot
# in accordance with what it sees.

import roslib
roslib.load_manifest("people_follow")
import sys, rospy, cv, cv2
from std_msgs.msg import String
from geometry_msgs.msg import Twist, Point, Point32, Polygon
from sensor_msgs.msg import Image, LaserScan, Joy
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import subprocess, signal
import math
import atexit

#----------------------------------------------------

class people_follow:
    def __init__(self, config):
        # load object properties from config dict
        self.__dict__.update(config)
        self.setup_ros_node()
        self.target = None
        self.gestures = []
        self.people = []

#----------------------------------------------------

    def setup_ros_node(self):
        rospy.init_node('people_follow')
        self.control_pub = rospy.Publisher(self.vel_topic, Twist)
        self.periodic_sub = rospy.Subscriber(self.gesture_topic, Polygon, self.handle_gesture)
        self.person_sub = rospy.Subscriber(self.person_topic, Polygon, self.handle_person)
        self.bridge = CvBridge()

#----------------------------------------------------

    def handle_gesture(self, polygon):
        rects = self.unpack_polygon(polygon.points)
        self.gestures = rects

        # TODO: do something based on this detected gesture

#----------------------------------------------------

    def handle_person(self, polygon):
        rects = self.unpack_polygon(polygon.points)
        self.people = rects

        # TODO: do something based on this detected person

#----------------------------------------------------

    def unpack_polygon(self, polygon):
        rectangles = []
        i = 0
        while i+2 < len(polygon):
            p0 = (polygon[i].x, polygon[i].y)
            p1 = (polygon[i+1].x, polygon[i+1].y)
            p2 = (polygon[i+2].x, polygon[i+2].y)
            p3 = (polygon[i+3].x, polygon[i+3].y)

            rectangles.append((p0,p1,p2,p3))

            i += 4

#----------------------------------------------------
    
    def publish_move(self, forward, turn):
        command = Twist()
        if abs(forward) > self.min_linear_speed:
            command.linear.x = forward
        if abs(turn) > self.min_angular_speed:
            command.angular.z = turn
        self.control_pub.publish(command)

#----------------------------------------------------
    
    def control_loop(self):
        rate = rospy.Rate(self.control_rate)
        while not rospy.is_shutdown():
            forward_amount, turn_amount = self.calculate_speed()
            self.publish_move(forward_amount, turn_amount)
            
            rate.sleep()

#----------------------------------------------------

    def ros_msg_to_cv2(self, ros_msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv(ros_msg, desired_encoding=self.image_encoding)
        except CvBridgeError, e:
            print e

        return np.asarray(cv_image[:,:])

#----------------------------------------------------
# END CLASS PEOPLE_FOLLOW
#----------------------------------------------------

if __name__ == "__main__":
    config = dict(
        min_linear_speed = rospy.get_param("~min_linear_speed", 0.1),
        max_linear_speed = rospy.get_param("~max_linear_speed", 0.95),
        min_angular_speed = rospy.get_param("~min_angular_speed", 0.1),
        max_angular_speed = rospy.get_param("~max_angular_speed", 0.95),
        control_rate = rospy.get_param("~control_rate", 30),
        detection_timeout = rospy.get_param("~detected_person_timeout", 3),
        gesture_topic = rospy.get_param("~gesture_topic", "periodic_gestures"),
        person_topic = rospy.get_param("~person_topic", "detected_people"),
        vel_topic = rospy.get_param("~cmd_vel_topic", "husky/plan_cmd_vel"))

    pf = people_follow(config)
    pf.control_loop()

