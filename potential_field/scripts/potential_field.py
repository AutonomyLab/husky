#! /usr/bin/env python

# This ROS node computes a potential field sum based on
# the laser scan readings, and publishes this sum for
# use in other nodes' obstacle avoidance logic

import roslib, sys, rospy, cv, cv2
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point
import numpy as np

#------------------------------------------

class potential_field:
    def __init__(self):
        rospy.init_node("potential_field")

        laser_topic = rospy.get_param("~laser_topic", "lidar/scan")
        potential_field_topic = rospy.get_param("~potential_field_topic", "potential_field_sum")

        self.sample_rate = rospy.get_param("~sample_rate", 5)

        self.min_angle = rospy.get_param("~min_angle", -np.pi/2)
        self.max_angle = rospy.get_param("~max_angle", np.pi/2)
        self.side_obstacle_force = rospy.get_param("~side_obstacle_force", 2.0)
        self.front_obstacle_force = rospy.get_param("~front_obstacle_force", 8.0)

        self.laser_sub = rospy.Subscriber(laser_topic, LaserScan, self.handle_laser)
        self.potential_field_pub = rospy.Publisher(potential_field_topic, Point)

        # laser readings
        self.laser = None

#------------------------------------------

    def start(self):
        rate = rospy.Rate(self.sample_rate)
        while not rospy.is_shutdown():
            self.compute_potential()
            rate.sleep()

#------------------------------------------

    def compute_potential(self):
        if self.laser == None:
            return

        laser_data = self.laser
        ranges = laser_data.ranges
        angle = laser_data.angle_min
        resolution = laser_data.angle_increment

        vector_sum = np.array([0.0, 0.0])

        for r in ranges:
            if (r < laser_data.range_min or
                    angle > self.max_angle or
                    angle < self.min_angle):
                distance = laser_data.range_max
            else:
                distance = r

            mag = 1.0 / (distance * distance) * self.compute_force(angle, laser_data.angle_max)
            vector = np.array([np.cos(angle + np.pi), np.sin(angle + np.pi)])
            vector *= mag

            vector_sum += vector
            
            angle += resolution

        vector_sum /= (len(ranges) + 1.0) # no division by zero

        self.publish_sum(vector_sum[0], vector_sum[1])


#------------------------------------------

    def handle_laser(self, laser_data):
        self.laser = laser_data
        
#------------------------------------------

    # make the force linear in the angle of the reading,
    # with the strongest forces from obstacles in front and
    # weakest from obstacles to the sides
    def compute_force(self, theta, max_theta):
        diff = abs(theta)
        normalized_diff = diff / max_theta
        slope = self.front_obstacle_force - self.side_obstacle_force
        return (1.0 - normalized_diff) * slope + self.side_obstacle_force

#------------------------------------------

    def publish_sum(self, x, y):
        vector = Point(x, y, 0)
        self.potential_field_pub.publish(vector)

#------------------------------------------

if __name__ == "__main__":
    pf = potential_field()
    pf.start()
