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
        self.detection_timeout = rospy.Duration(self.detection_timeout)
        self.setup_ros_node()
        self.target = None
        self.last_detection = rospy.Time.now()
        self.gestures = []
        self.people = []

#----------------------------------------------------

    def setup_ros_node(self):
        rospy.init_node('people_follow')
        self.control_pub = rospy.Publisher(self.vel_topic, Twist)
        self.subscribe_node("gestures")
        self.subscribe_node("people")
        self.bridge = CvBridge()

#----------------------------------------------------

    def unsubscribe_node(self, target):
        if target == "gestures":
            self.periodic_sub.unregister()
            self.peridic_sub = None
        elif target == "people":
            self.person_sub.unregister()
            self.person_sub = None

#----------------------------------------------------

    def subscribe_node(self, target):
        if target == "gestures":
            self.periodic_sub = rospy.Subscriber(self.gesture_topic, Polygon, self.handle_gestures)
        elif target == "people":
            self.person_sub = rospy.Subscriber(self.person_topic, Polygon, self.handle_people)

#----------------------------------------------------
    
    def toggle_node(self, target):
        if target == "gestures":
            if self.periodic_sub == None:
                self.subscribe(target)
            else:
                self.unsubscribe(target)
        elif target == "people":
            if self.person_sub == None:
                self.subscribe(target)
            else:
                self.unsubscribe(target)

#----------------------------------------------------

    def handle_gestures(self, polygon):
        self.gestures = self.unpack_polygon(polygon.points)

#----------------------------------------------------

    def handle_people(self, polygon):
        self.people = self.unpack_polygon(polygon.points)

#----------------------------------------------------

    def unpack_polygon(self, polygon):
        # for normalizing the objects' positions
        w = float(self.frame_width)
        h = float(self.frame_height)

        rectangles = []
        i = 0
        while i+2 < len(polygon):
            p0 = (polygon[i].x/w, polygon[i].y/h)
            p1 = (polygon[i+1].x/w, polygon[i+1].y/h)
            p2 = (polygon[i+2].x/w, polygon[i+2].y/h)
            p3 = (polygon[i+3].x/w, polygon[i+3].y/h)

            rectangles.append((p0,p1,p2,p3))

            i += 4

        return rectangles

#----------------------------------------------------
    
    def publish_move(self, forward, turn):
        command = Twist()
        command.linear.x = forward
        command.angular.z = turn
        self.control_pub.publish(command)

#----------------------------------------------------

    def compute_vel(self):
        if len(self.people) > 0:
            # find the person closest to the center
            best_distance = sys.maxint
            best_person = None
            for person in self.people:
                dist = abs((person[0][0]+person[1][0])/2 - 0.5)
                if dist < best_distance:
                    best_distance = dist
                    best_person = person

            self.target = (best_person[0][0]+best_person[1][0]) / 2

        elif len(self.gestures) > 0:
            alpha = 1
            beta = 0.01

            best_score = 0
            best_gesture = None
            for gesture in self.gestures:
                dist = abs((gesture[0][0]+gesture[1][0])/2 - 0.5)
                area = (gesture[1][0]-gesture[0][0])*(gesture[2][1]-gesture[1][1])
                score = alpha*dist + beta*area

                if score > best_score:
                    best_gesture = gesture

            self.target = (best_gesture[0][0]+best_gesture[1][0]) / 2

        elif rospy.Time.now() - self.last_detection > self.detection_timeout:
            self.target = None

        if self.target == None:
            return (0,0)
        else:
            linear = -2*self.target + 1
            angular = 0.5 - self.target*2
            
            return (linear, angular)           

#----------------------------------------------------
    
    def control_loop(self):
        rate = rospy.Rate(self.control_rate)
        while not rospy.is_shutdown():
            forward_amount, turn_amount = self.compute_vel()
            if forward_amount != 0 or turn_amount != 0:
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
        frame_width = rospy.get_param("~frame_width", 640),
        frame_height = rospy.get_param("~frame_height", 480),
        control_rate = rospy.get_param("~control_rate", 30),
        detection_timeout = rospy.get_param("~detected_person_timeout", 3),
        gesture_topic = rospy.get_param("~gesture_topic", "periodic_gestures/gestures"),
        person_topic = rospy.get_param("~person_topic", "person_detection/people"),
        vel_topic = rospy.get_param("~cmd_vel_topic", "husky/plan_cmd_vel"))

    pf = people_follow(config)
    pf.control_loop()

