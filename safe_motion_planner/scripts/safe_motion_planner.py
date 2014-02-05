#! /usr/bin/env python

# This node interprets messages from the joy topic
# and applies the potential field approach to create
# a safe navigation vector from the joystick command
# and the surrounding obstacles

#-------------------------------------------------

# NODE PARAMETERS:

# ~joy_topic                "joy"
# ~potential_field_topic    "potential_field_sum"
# ~cmd_vel_topic            "husky/cmd_vel"
# ~plan_cmd_vel_topic       "husky/plan_cmd_vel"

# ~joy_vector_magnitude     1.5
# ~drive_scale              1.0
# ~turn_scale               1.0
# ~safe_reverse_speed       0.2
# ~cmd_rate                 10.0

# ~deadman_button           0 (X)
# ~planner_button           3 (Y)

#-------------------------------------------------

import roslib, sys, rospy
from sensor_msgs.msg import LaserScan, Joy
from geometry_msgs.msg import Point, Twist
import numpy as np

#-------------------------------------------------

class safe_teleop:
    def __init__(self):
        rospy.init_node("safe_teleop")

        joy_topic = rospy.get_param("~joy_topic", "joy")
        potential_field_topic = rospy.get_param("~potential_field_topic", "potential_field_sum")
        cmd_vel_topic = rospy.get_param("~cmd_vel_topic", "husky/cmd_vel")
        plan_cmd_vel_topic = rospy.get_param("~plan_cmd_vel_topic", "husky/plan_cmd_vel")

        rospy.Subscriber(joy_topic, Joy, self.handle_joy)
        rospy.Subscriber(potential_field_topic, Point, self.handle_potential_field)
        rospy.Subscriber(plan_cmd_vel_topic, Twist, self.handle_plan)

        self.cmd_pub = rospy.Publisher(cmd_vel_topic, Twist)

        self.obstacle_vector = None
        self.joy_vector = None

        self.magnitude = rospy.get_param("~joy_vector_magnitude", 1.5)
        self.drive_scale = rospy.get_param("~drive_scale", 1)
        self.turn_scale = rospy.get_param("~turn_scale", 1)
        self.safe_reverse_speed = rospy.get_param("~safe_reverse_speed", 0.2)

        self.joy_data = None

        # X = 0
        # A = 1
        # B = 2
        # Y = 3
        # LBump = 4
        # RBump = 5
        self.deadman_button = rospy.get_param("~deadman_button", 0)
        self.planner_button = rospy.get_param("~planner_button", 3)
        self.override_buttons = [self.deadman_button, 4, 5]

        self.safe_motion = False
        self.override = False
        self.planned_motion = False

        self.planned_motion_time = rospy.Time.now()

        self.planner_cmd = Twist()

#-------------------------------------------------

    def start(self):
        rate = rospy.Rate(rospy.get_param("~cmd_rate", 10))
        while not rospy.is_shutdown():
            # expire old planner commands
            if rospy.Time.now() - self.planned_motion_time > 0.1:
                self.planner_cmd = Twist()
            cmd = self.compute_motion_cmd()
            if cmd != None:
                self.cmd_pub.publish(cmd)
            rate.sleep()

#-------------------------------------------------

    # take the joystick vector and obstacle vector,
    # and sum them together to get a desired motion vector,
    # then create a motion command that corresponds to this
    def compute_motion_cmd(self):
        cmd = None

        if self.joy_data == None or self.obstacle_vector == None:
            cmd = None

        elif self.planned_motion and self.planner_cmd == None:
            cmd = None
 
        elif self.planned_motion:
            cmd = self.planner_cmd

            # convert planned command to direction vector
            # and add it to obstacle vector to find resulting
            # safe motion.

            cmd_vector = np.asarray([cmd.linear.x, -cmd.angular.z])
            cmd_vector /= np.linalg.norm(cmd_vector)
            cmd_vector *= self.magnitude

            vector_sum = cmd_vector + self.obstacle_vector
            vector_sum /= np.linalg.norm(vector_sum)

            # we can't see backward, so restrict backward motion
            if joy_cmd_vector[0] >= 0:
                vector_sum[0] = max(0, vector_sum[0])
            else:
                vector_sum[0] = max(-self.safe_reverse_speed, vector_sum[0])
            # convert the resultant vector into a
            # linear and angular velocity for moving the robot

            cmd = Twist()
            cmd.linear.x = vector_sum[0] * self.drive_scale
            cmd.angular.z = vector_sum[1] * -self.turn_scale
   
        # don't move if we're not touching the thumb stick
        elif self.joy_data.axes[1] == 0.0 and self.joy_data.axes[0] == 0.0:
            cmd = None

        elif self.override:
            cmd = Twist()
            cmd.linear.x = self.joy_data.axes[1] * self.drive_scale
            cmd.angular.z = self.joy_data.axes[0] * self.turn_scale

        elif self.safe_motion:
            vector_sum = self.joy_vector + self.obstacle_vector
            vector_sum /= np.linalg.norm(vector_sum)

            # multiply by the norm of the joystick command,
            # so we only move a little when the axes are only
            # slightly pressed

            joy_cmd_vector = np.array([self.joy_data.axes[1], self.joy_data.axes[0]])
            vector_sum *= np.linalg.norm(joy_cmd_vector)

            # we can't see backward, so restrict backward motion
            if joy_cmd_vector[0] >= 0:
                vector_sum[0] = max(0, vector_sum[0])
            else:
                vector_sum[0] = max(-self.safe_reverse_speed, vector_sum[0])

            # convert the resultant vector into a
            # linear and angular velocity for moving the robot

            cmd = Twist()
            cmd.linear.x = vector_sum[0] * self.drive_scale
            cmd.angular.z = vector_sum[1] * -self.turn_scale
        return cmd

#-------------------------------------------------

    def handle_joy(self, joy_data):
        self.joy_data = joy_data

        self.override = True
        for button in self.override_buttons:
            if joy_data.buttons[button] == 0:
                self.override = False

        self.safe_motion = (not self.override) and joy_data.buttons[self.deadman_button] != 0
        self.planned_motion = (not self.override) and (not self.safe_motion) and joy_data.buttons[self.planner_button] != 0

        x = joy_data.axes[1]
        y = -joy_data.axes[0]
        joy_vector = np.array([x, y])
        joy_vector /= np.linalg.norm(joy_vector)
        joy_vector *= self.magnitude

        self.joy_vector = joy_vector

#-------------------------------------------------

    def handle_potential_field(self, potential_field):
        self.obstacle_vector = np.array([potential_field.x, potential_field.y])

#-------------------------------------------------

    def handle_plan(self, data):
        self.planner_cmd = data
        self.planned_motion_time = rospy.Time.now()

#-------------------------------------------------

if __name__ == "__main__":
    st = safe_teleop()
    st.start()
