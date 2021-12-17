#!/usr/bin/env python
import numpy as np

import message_filters
import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2 as pc2
from spencer_tracking_msgs.msg import DetectedPersons, DetectedPerson

from tfpose_ros.msg import Persons


class PoseForSpencer(object):
    def __init__(self):
        # parameters
        self.__pub_pose = rospy.Publisher('~detected_persons', DetectedPersons, queue_size=1)

        self.__id = 0
        points_sub = message_filters.Subscriber("~points", PointCloud2)
        persons_sub = message_filters.Subscriber("~persons", Persons)
        sub = message_filters.ApproximateTimeSynchronizer([points_sub, persons_sub], 10, 0.5)
        sub.registerCallback(self.__callback)

    def __persons_to_msg(self, persons_msg, points):

        persons = DetectedPersons()
        height, width = points.height, points.width
        for p in persons_msg.persons:
            for k in p.body_part:
                if k.part_id in [1, 8, 11]:
                    x, y = int(k.x * width + 0.5), int(k.y * height + 0.5)
                    centers = [[xx, yy] for yy in range(max(y - 3, 0), min(y + 3, height))
                               for xx in range(max(x - 3, 0), min(x + 3, width))]
                    pts = [p for p in pc2.read_points(points, ('x', 'y', 'z'), uvs=centers, skip_nans=True)]
                    if not pts:
                        continue
                    pt = np.mean(pts, axis=0)
                    person = DetectedPerson()
                    person.modality = "Pose"
                    person.pose.pose.position.x = pt[0]
                    person.pose.pose.position.y = pt[1]
                    person.pose.pose.position.z = pt[2]
                    person.confidence = k.confidence
                    person.detection_id = self.__id
                    self.__id += 1
                    large_var = 999999999
                    pose_variance = 0.05
                    person.pose.covariance[0 * 6 + 0] = pose_variance
                    person.pose.covariance[1 * 6 + 1] = pose_variance
                    person.pose.covariance[2 * 6 + 2] = pose_variance
                    person.pose.covariance[3 * 6 + 3] = large_var
                    person.pose.covariance[4 * 6 + 4] = large_var
                    person.pose.covariance[5 * 6 + 5] = large_var
                    persons.detections.append(person)
                    break

        return persons

    def __callback(self, points, persons_msg):
        msg = self.__persons_to_msg(persons_msg, points)
        msg.header = points.header

        self.__pub_pose.publish(msg)


if __name__ == '__main__':
    rospy.init_node('pose_for_spencer')
    node = PoseForSpencer()
    rospy.spin()
