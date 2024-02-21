#!/usr/bin/env python
import cv2
import numpy as np
import rospy
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PointStamped

import random

filtered_data = None

class Object:
    def __init__(self, X, Y, time=1 / 24):
        self.x = X
        self.y = Y
        self.t = time

    def update(self):
        self.x += random.choice([-1, 1]) * self.x * self.t
        self.y += random.choice([-1, 1]) * self.y * self.t

    def get(self):
        return int(self.x), int(self.y)

global obj

obj = Object(250, 250)

map_env = np.zeros((500, 500))


def callback(data):
    global pub, map_env, filtered_data
    filtered_data = data.point
    cv2.circle(map_env, (int(filtered_data.y), int(filtered_data.x)), 10, (255, 255, 255), thickness=1)
    

def publisher():
    global map_env, obj
    rospy.init_node('Sensor_node', anonymous=False)
    pub = rospy.Publisher('sensor_data', Float64MultiArray, queue_size=15)
    rospy.Subscriber('filtered_position', PointStamped, callback)
    rate = rospy.Rate(10)  # 1 Hz
    while not rospy.is_shutdown():
        obj.update()
        measurement = obj.get()
        

        if 0 < measurement[0] < 500 and 0 < measurement[1] < 500:
            
            map_env[measurement] = 255
            cv2.imshow('measurements', map_env)
            rospy.loginfo("Published Array: \n%s", measurement)
            cv2.waitKey(10)
            map_env = np.zeros((500, 500))
        else:
            map_env = np.zeros((500, 500))
            obj = Object(250, 250)
            
        msg_pub = Float64MultiArray()
        msg_pub.data = measurement

            # Publish the message
        pub.publish(msg_pub)
        rate.sleep()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        publisher()
    except rospy.ROSInterruptException:
        pass
