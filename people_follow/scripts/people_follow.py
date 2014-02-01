#! /usr/bin/env python

# This ROS node listens for periodic motion detection and
# HOG person detection and controls the behavior of the robot
# in accordance with what it sees.

import roslib
roslib.load_manifest("people_follow")
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

class people_follow:
    def __init__(self, config):
        self.setup_ros_node(config)

        self.target = None

        # load object properties from config dict
        self.__dict__.update(config)

        self.autonomy_enabled = False

#----------------------------------------------------

    def setup_ros_node(self, config):
        rospy.init_node('people_follow')
        #self.republisher = subprocess.Popen("rosrun image_transport republish compressed in:=/axis/image_raw raw out:=/axis/image_raw/decompressed", shell=True)

        self.control_pub = rospy.Publisher(config.vel_topic, Twist)
        
        self.periodic_sub = rospy.Subscriber(config.gesture_topic, Polygon, self.handle_gesture)
        self.person_sub = rospy.Subscriber(config.person_topic, Polygon, self.handle_person)

        self.bridge = CvBridge()

#----------------------------------------------------

    def handle_gesture(self, polygon):
        # TODO: polygon is a list of points, each group of 4
        # representing the bounding rectangle of a detected gesture
        pass

#----------------------------------------------------

    def handle_person(self, polygon):
        # TODO: polygon is a list of points, each group of 4
        # representing the bounding rectangle of a detected person
        pass

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
        max_person_size = 1.0,
        turning_rate = 0.5,
        max_speed = 0.95,
        min_linear_speed = 0.1,
        min_angular_speed = 0.1,
        control_rate = 30,
        detection_timeout = 3.0,
        gesture_topic = rospy.get_param("~gesture_topic", "periodic_gestures"),
        person_topic = rospy.get_param("~person_topic", "detected_people"),
        vel_topic = rospy.get_param("~cmd_vel_topic", "husky/plan_cmd_vel"),
        debug = False)

    pf = people_follow(config)
    pf.control_loop()

