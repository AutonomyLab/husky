#! /usr/bin/env python

# This ROS node watches the camera and applies the HOG people
# detection algorithm to find pedestrians.

import roslib
roslib.load_manifest("people_follow")
import sys, rospy, cv, cv2
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, LaserScan, Joy
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import subprocess, signal
import math
import atexit

#----------------------------------------------------

class people_follow:
    def __init__(self, config):
        self.setup_ros_node()

        self.target = None

        # load object properties from config dict
        self.__dict__.update(config)

        self.autonomy_enabled = False
        self.person_detection_interval = rospy.Duration(0.75)
        self.person_detection_last = rospy.Time.now()
        self.last_detection = rospy.Time.now()
        # needs to be a Duration object
        self.detection_timeout = rospy.Duration(config["detection_timeout"])
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        # store the laser range readings here
        if self.debug:
            cv2.namedWindow("Camera Feed")

#----------------------------------------------------

    def setup_ros_node(self):
        rospy.init_node('people_follow', anonymous=True)
        self.republisher = subprocess.Popen("rosrun image_transport republish compressed in:=/axis/image_raw raw out:=/axis/image_raw/decompressed", shell=True)
        self.control_pub = rospy.Publisher("husky/cmd_vel", Twist)
        self.image_sub = rospy.Subscriber("axis/image_raw/decompressed", Image, self.handle_image)
        self.laser_sub = rospy.Subscriber("lidar/scan", LaserScan, self.handle_laser)
        self.joy_sub = rospy.Subscriber("joy", Joy, self.handle_joy)
        self.bridge = CvBridge()

#----------------------------------------------------

    def handle_image(self, data):
        if rospy.Time.now() - self.person_detection_last < self.person_detection_interval:
            return # don't do anything unless enough time has elapsed

        begin_processing = rospy.Time.now()
        self.person_detection_last = rospy.Time.now()
        cv2_image = self.ros_msg_to_cv2(data)
        # try to detect people and set the servoing target
        self.process_image(cv2_image)
        elapsed = rospy.Time.now() - begin_processing
        # adjust frame processing rate to match detector rate,
        # plus a small margin
        self.person_detection_interval = rospy.Duration(elapsed.to_sec() + 0.1)

#----------------------------------------------------

    def handle_joy(self, data):
        old = self.autonomy_enabled
        self.autonomy_enabled = data.buttons[self.autonomy_button] == 1

        # print if autonomy changed
        if old != self.autonomy_enabled:
            print "autonomy_enabled: %s" % self.autonomy_enabled

#----------------------------------------------------

    def handle_laser(self, data):
        self.ranges = data.ranges
        self.range_min = data.range_min
        self.range_max = data.range_max
        self.angle_min = data.angle_min
        self.angle_max = data.angle_max
        self.angle_increment = data.angle_increment

#----------------------------------------------------

    def detect_people(self, cv2_image):
        before = rospy.Time.now()

        people_found, weights = self.hog.detectMultiScale(
                cv2_image, 
                winStride=self.hog_win_stride, 
                padding=self.hog_padding, 
                scale=self.hog_scale)

        elapsed = rospy.Time.now() - before

        return people_found

#----------------------------------------------------

    def person_too_close(self, person):
        if person == None:
            return False

        person_height = float(person[3])
        return (person_height / self.camera_resolution_y) > self.max_person_size

#----------------------------------------------------

    def process_image(self, cv2_image):
        people_found = self.detect_people(cv2_image)

        # determine which person to track, and save their center
        # location / max_width in self.target 

        # for now we're just servoing to the center target

        height, width, depth = cv2_image.shape
        width_divisor = float(width)
        middle_person = None
        middle_loc = None
        person_height = None
        for person in people_found:
            # if we're doing fixed window size detection,
            # we'll be missing the width/height info
            if len(person) < 4:
                person = [person[0], person[1], 64, 128]
            pt1 = (person[0], person[1])
            pt2 = (pt1[0] + person[2], pt1[1] + person[3])

            # find the person closest to the center of the view
            person_center = pt1[0] + person[2]/2
            person_center /= width_divisor
            if middle_loc == None or abs(middle_loc - 0.5) > abs(person_center-0.5):
                middle_loc = person_center
                middle_person = person

            if self.debug:
                cv2.rectangle(cv2_image, pt1, pt2, (255,255,255))

        # set our servoing target
        if middle_loc != None:
            self.last_detection = rospy.Time.now()
            self.target = middle_loc

        if self.debug:
            cv2.imshow("Camera Feed", cv2_image)
            cv2.waitKey(1)

        if rospy.Time.now() - self.last_detection > self.detection_timeout:
            self.target = None
        if self.person_too_close(middle_person):
            self.target = None

#----------------------------------------------------
    
    def publish_move(self, forward, turn):
        command = Twist()
        if abs(forward) > self.min_linear_speed:
            command.linear.x = forward
        if abs(turn) > self.min_angular_speed:
            command.angular.z = turn
        self.control_pub.publish(command)

#----------------------------------------------------

    def obstacle_inside_safety_box(self):
        angle = self.angle_min
        dangerCount = 0
        for r in self.ranges:
            if r > self.range_min:
                x, y = angle_mag_to_2d_vector(angle, r)
                if (abs(y) < (self.emergency_stop_box_width/2) and
                        abs(x) < self.emergency_stop_box_depth):
                    dangerCount += 1
            angle += self.angle_increment

        return dangerCount >= self.min_laser_readings

#----------------------------------------------------

    def calculate_speed(self):
        # use potential field sum vector to decide
        # which direction to move
        vector = self.calculate_potential_field_sum()
        # don't go backward
        forward_amount = max(0, vector[0] * self.max_speed)
        turn_amount = -vector[1] * self.turning_rate

        if self.obstacle_inside_safety_box():
            forward_amount = 0
            turn_amount = 0

        return (forward_amount, turn_amount)

#----------------------------------------------------

    # return a weight to multiply the repulsive force by,
    # based on the angle the laser reading is at.
    # This is intended to weight obstacles in front of the
    # robot more heavily than objects to the side of the robot.
    def weight_obstacle(self, angle):
        diff = abs(angle)
        max_diff = self.angle_max
        normalized_diff = diff / max_diff
        slope = self.max_obstacle_weight - self.min_obstacle_weight
        return (1.0 - normalized_diff) * slope + self.min_obstacle_weight

#----------------------------------------------------

    def calculate_potential_field_sum(self):
        if self.target == None:
            user_vector = np.array([0.0, 0.0])
        else:
            heading = -np.pi/2 + self.target * np.pi
            magnitude = self.user_attract
            user_vector = angle_mag_to_2d_vector(heading, magnitude)

        obstacle_vector_sum = np.array([0.0, 0.0])
        angle = self.angle_min
        resolution = self.angle_increment
        count = 0.0
        for r in self.ranges:
            if (r < self.range_min or
                    angle > self.potential_field_max_angle or
                    angle < self.potential_field_min_angle):
                distance = self.range_max
            else:
                distance = r
            mag = 1.0 / (distance * distance) * self.weight_obstacle(angle)
            vector = angle_mag_to_2d_vector(angle + np.pi, mag)
            obstacle_vector_sum += vector
            angle += resolution
            count += 1

        # normalize a bit
        obstacle_vector_sum /= count

        #print "Obstacle vector: %s" % obstacle_vector_sum
        #print "User vector: %s" % user_vector
        #print "Resultant vector: %s" % (user_vector + obstacle_vector_sum)

        return user_vector + obstacle_vector_sum

#----------------------------------------------------
    
    def control_loop(self):
        rate = rospy.Rate(self.control_rate)
        while not rospy.is_shutdown():
            if self.autonomy_enabled:
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

    def cleanup(self):
        self.republisher.kill()

#----------------------------------------------------
# END CLASS PEOPLE_FOLLOW
#----------------------------------------------------

# angle is in radians
def angle_mag_to_2d_vector(angle, mag):
    vector = np.array([np.cos(angle), np.sin(angle)])
    vector *= mag
    return vector

#----------------------------------------------------

def main(args):
    debug = "-d" in sys.argv

    config = dict(
        max_person_size = 1.0,
        turning_rate = 0.5,
        emergency_stop_box_width = 0.75,
        emergency_stop_box_depth = 0.5,
        # must be 5 danger readings to be sure we're close to an obstacle
        min_laser_readings = 5, 
        potential_field_min_angle = -np.pi/2,
        potential_field_max_angle = np.pi/2,
        max_speed = 1.0,
        min_linear_speed = 0.1,
        min_angular_speed = 0.1,
        min_obstacle_weight = 1.5,
        max_obstacle_weight = 6.0,
        user_attract = 1.0,
        control_rate = 30,
        autonomy_button = 3,
        detection_timeout = 3.0,
        hog_win_stride = (8,8),
        hog_padding = (2,2),
        hog_scale = 1.075,
        camera_resolution_x = 640,
        camera_resolution_y = 480,
        image_encoding = "bgr8",
        debug = debug)

    pf = people_follow(config)
    atexit.register(pf.cleanup)
    pf.control_loop()
    
#----------------------------------------------------

if __name__ == "__main__":
    main(sys.argv)
