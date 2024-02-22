#!/usr/bin/env python
import random

import rospy
import numpy as np
from object_tracking_EKF.msg import MeasurementMsg, StateMsg
import cv2


class MeasurementSimulator:
    def __init__(self):
        # Subscribe to measurement topic
        self.subscriber = rospy.Subscriber('estimated_state_topic', StateMsg, self.filtered_data_callback)

        rospy.init_node('Sensor_node', anonymous=False)

        # Parameters
        self.filtered_position_x = None
        self.filtered_position_y = None
        self.range_measurement = None
        self.bearing_measurement = None
        self.map_env = np.zeros((500, 500))
        self.sensor_position = np.array([0.0, 0.0])  # Sensor position
        self.object_velocity = np.array([0.0, 0.0])  # Object velocity (initial)
        self.acceleration = np.array([0.0, 0.0])
        self.object_position = np.array([0.0, 0.0])

        # Publisher for measurement
        self.measurement_pub = rospy.Publisher('measurement_topic', MeasurementMsg, queue_size=15)

        # Simulation loop rate (Hz)
        self.rate = rospy.Rate(10)

        self.simulate_measurements()

    def filtered_data_callback(self, msg):
        self.filtered_position_x = msg.x
        self.filtered_position_y = msg.y

    def simulate_measurements(self):
        while not rospy.is_shutdown():
            # Update object position based on velocity
            t = self.rate.sleep_dur.to_sec()
            self.object_velocity += t * self.acceleration
            self.object_position += t * self.object_velocity + 0.5 * (t ** 2) * self.acceleration

            # Simulate measurement
            self.range_measurement = np.linalg.norm(self.object_position - self.sensor_position)
            self.bearing_measurement = np.arctan2(self.object_position[1] - self.sensor_position[1],
                                                  self.object_position[0] - self.sensor_position[0])

            # Publish measurement
            measurement_msg = MeasurementMsg()
            measurement_msg.range = self.range_measurement
            measurement_msg.bearing = self.bearing_measurement
            self.measurement_pub.publish(measurement_msg)
            rospy.loginfo("range_measurement: \n%s", self.range_measurement)
            rospy.loginfo("bearing_measurement: \n%s", self.bearing_measurement)

            x_obj = self.range_measurement * np.cos(self.bearing_measurement)
            y_obj = self.range_measurement * np.sin(self.bearing_measurement)
            if 0 < x_obj < 500 and 0 < y_obj < 500:
                self.map_env[int(y_obj), int(x_obj)] = 255

                if self.filtered_position_y is not None and self.filtered_position_x is not None:
                    cv2.circle(self.map_env,
                               (int(self.filtered_position_y), int(self.filtered_position_x)),
                               7, (255, 255, 255))
                cv2.imshow('EKF object tracker', self.map_env)
                cv2.waitKey(10)
            else:
                self.sensor_position = np.array([0.0, 0.0])  # Sensor position
                self.object_velocity = np.array([0.0, 0.0])  # Object velocity (initial)
                self.object_position = np.array([0.0, 0.0])

            self.acceleration = np.array((0.1, 0.1))  # np.array((random.random(), random.random()))
            self.map_env = np.zeros((500, 500))

            # Sleep to maintain the loop rate
            self.rate.sleep()

    def spin(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        measurement_simulator = MeasurementSimulator()
        measurement_simulator.spin()
    except rospy.ROSInterruptException:
        pass
