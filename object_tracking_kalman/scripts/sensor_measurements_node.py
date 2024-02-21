#!/usr/bin/env python
import cv2
import numpy as np
import rospy
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PointStamped

import random


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

global map_env


def callback(data):
    global pub, map_env
    filtered_data = data.point
    cv2.rectangle(map_env, (int(filtered_data.y) - 10, int(filtered_data.x) - 10),
                  (int(filtered_data.y + 10), int(filtered_data.x + 10)),
                  (255, 255, 255), thickness=1)

    print(filtered_data.x, filtered_data.y)


def publisher():
    global map_env, obj
    rospy.init_node('array_publisher', anonymous=True)
    pub = rospy.Publisher('sensor_data', Float64MultiArray, queue_size=15)
    rospy.Subscriber('filtered_position', PointStamped, callback)
    rate = rospy.Rate(10)  # 1 Hz
    while not rospy.is_shutdown():
        obj.update()
        measurement = obj.get()
        if 0 < measurement[0] < 500 and 0 < measurement[1] < 500:
            map_env = np.zeros((500, 500))
            map_env[measurement] = 255
            cv2.imshow('measurements', map_env)
            msg_pub = Float64MultiArray()
            msg_pub.data = measurement

            # Publish the message
            pub.publish(msg_pub)

            rospy.loginfo("Published Array: \n%s", measurement)

            cv2.waitKey(10)
        else:
            obj = Object(250, 250)

        rate.sleep()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        publisher()
    except rospy.ROSInterruptException:
        pass
