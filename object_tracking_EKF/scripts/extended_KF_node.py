#!/usr/bin/env python

import rospy
import numpy as np
from object_tracking_EKF.msg import StateMsg, MeasurementMsg
from math import atan2, sqrt


class EKFNode:
    def __init__(self):
        self.G = None
        self.H = None
        rospy.init_node('ekf_node', anonymous=False)

        # Define state vector [x, y, vx, vy]
        self.state = np.random.random((4, 1))

        # Define covariance matrix
        self.P = np.eye(4)

        # Define process noise covariance matrix
        self.R = np.eye(4) * 0.01  # Adjust according to your system

        # Define measurement noise covariance matrix
        self.Q = np.eye(2) * 0.1  # Adjust according to your system

        # Publisher for estimated state
        self.state_pub = rospy.Publisher('estimated_state_topic', StateMsg, queue_size=15)

        self.dt = 0.1

        # Subscribe to measurement topic
        rospy.Subscriber('measurement_topic', MeasurementMsg, self.measurement_callback)

        rospy.spin()

    def measurement_callback(self, msg):
        # Extract measurement data
        z = np.array([[msg.range], [msg.bearing]])

        # Prediction step
        self.predict()

        # Update step
        self.update(z)

        # Publish estimated state
        estimated_state_msg = StateMsg()
        estimated_state_msg.x = self.state[0, 0]
        estimated_state_msg.y = self.state[1, 0]
        estimated_state_msg.vx = self.state[2, 0]
        estimated_state_msg.vy = self.state[3, 0]
        range_ = sqrt((self.state[0, 0] - 0) ** 2 + (self.state[1, 0] - 0) ** 2)
        rospy.loginfo("filtered_position: \n%s", self.state)
        rospy.loginfo("Range: \n%s", range_)
        self.state_pub.publish(estimated_state_msg)

    def predict(self):
        # Nonlinear motion model: x = x + vx*dt, y = y + vy*dt

        self.state[0] += self.state[2] * self.dt
        self.state[1] += self.state[3] * self.dt

        # Jacobian matrix for the motion model
        self.G = np.array([[1, 0, self.dt, 0],
                           [0, 1, 0, self.dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

        # Propagate covariance matrix
        self.P = np.dot(np.dot(self.G, self.P), self.G.T) + self.R

    def update(self, z):
        # Measurement model: h(x) = [sqrt(x^2 + y^2), atan2(y, x)]
        hx = np.array([[sqrt(self.state[0, 0] ** 2 + self.state[1, 0] ** 2)],
                       [atan2(self.state[1, 0], self.state[0, 0])]])

        # Jacobian matrix for the measurement model
        self.H = np.array([[self.state[0, 0] / sqrt(self.state[0, 0] ** 2 + self.state[1, 0] ** 2),
                            self.state[1, 0] / sqrt(self.state[0, 0] ** 2 + self.state[1, 0] ** 2), 0, 0],
                           [-self.state[1, 0] / (self.state[0, 0] ** 2 + self.state[1, 0] ** 2),
                            self.state[0, 0] / (self.state[0, 0] ** 2 + self.state[1, 0] ** 2), 0, 0]])

        # Compute Kalman gain
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.Q
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # Update state estimate
        self.state += np.dot(K, (z - hx))

        # Update covariance matrix
        self.P = np.dot((np.eye(4) - np.dot(K, self.H)), self.P)


if __name__ == '__main__':
    try:
        ekf_node = EKFNode()
    except rospy.ROSInterruptException:
        pass
