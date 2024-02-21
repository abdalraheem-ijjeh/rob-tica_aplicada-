#!/usr/bin/env python

import rospy
import numpy as np
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float64MultiArray


class KalmanFilterNode2D:
    def __init__(self):
        rospy.init_node('kalman_filter_node_2d', anonymous=True)
        self.pub_filtered_position = rospy.Publisher('filtered_position', PointStamped, queue_size=10)
        self.sub_sensor_data = rospy.Subscriber('sensor_data', Float64MultiArray, self.sensor_data_callback)

        self.dt = 0.1  # Time step
        self.A = np.array([[1, 0, self.dt, 0],
                           [0, 1, 0, self.dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])  # State transition matrix
        self.B = np.zeros((4, 1))  # Control input matrix
        self.C = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])  # Measurement matrix
        self.R = np.eye(4) * 0.01  # Process noise covariance
        self.Q = np.eye(2) * 0.1  # Measurement noise covariance

        self.x = np.zeros((4, 1))  # Initial state estimate
        self.P = np.eye(4)  # Initial state covariance

    def predict(self):
        # Predict the next state
        self.x = np.dot(self.A, self.x)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.R

    def update(self, z):
        # Update the state estimate based on measurement
        y = z - np.dot(self.C, self.x)
        S = np.dot(np.dot(self.C, self.P), self.C.T) + self.Q
        K = np.dot(np.dot(self.P, self.C.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, self.C), self.P)

    def sensor_data_callback(self, data):
        z = np.array(data.data).reshape((2, 1))  # Convert sensor data to numpy array
        self.predict()
        self.update(z)

        # Publish filtered position
        filtered_position = PointStamped()
        filtered_position.header.stamp = rospy.Time.now()
        filtered_position.point.x = self.x[0, 0]
        filtered_position.point.y = self.x[1, 0]
        filtered_position.point.z = 0  # Assuming 2D motion, set z to 0
        self.pub_filtered_position.publish(filtered_position)
        rospy.loginfo("filtered_position: \n%s", self.x[0:2])


if __name__ == '__main__':
    try:
        kf_node = KalmanFilterNode2D()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
