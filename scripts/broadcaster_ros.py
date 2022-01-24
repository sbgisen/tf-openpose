#!/usr/bin/env python
import numpy as np
import tensorflow as tf
from threading import Lock

from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Point
import message_filters
import rospkg
import rospy
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs.point_cloud2 as pc2
from spencer_tracking_msgs.msg import DetectedPersons, DetectedPerson
from std_msgs.msg import ColorRGBA
from std_srvs.srv import SetBool, SetBoolResponse
from visualization_msgs.msg import MarkerArray, Marker

from tfpose_ros.estimator import TfPoseEstimator
from tfpose_ros.common import CocoPairsRender, CocoColors
from tfpose_ros.msg import Persons, Person, BodyPartElm
from tfpose_ros.networks import model_wh


class PoseEstimator(object):
    def __init__(self):
        # parameters
        resolution = rospy.get_param('~resolution', '432x368')
        self.__resize_out_ratio = float(rospy.get_param('~resize_out_ratio', '4.0'))
        self.__hz = rospy.get_param('~hertz', 5)
        self.__tf_lock = Lock()
        self.__prev_time = rospy.Time.now()
        self.__detection_id_increment = rospy.get_param('~detection_id_increment', 1)
        self.__last_detection_id = rospy.get_param('~detection_id_offset', 0)
        self.__visualize = rospy.get_param('~visualize', False)

        self.__graph_path = None
        try:
            self.__target_size = model_wh(resolution)
            ros_pack = rospkg.RosPack()
            package_path = ros_pack.get_path('tfpose_ros')
            self.__graph_path = rospy.get_param('~model', package_path + '/models/graph/cmu/graph_opt.pb')
        except Exception as e:
            rospy.logerr('invalid model: %s, e=%s' % (self.__graph_path, e))
            exit()

        self.__tf_config = tf.ConfigProto()
        self.__tf_config.gpu_options.per_process_gpu_memory_fraction = 0.25
        self.__tf_config.gpu_options.visible_device_list = "0"
        self.__pose_estimator = None

        self.__is_active = False
        self.__restart = False

        rospy.Service('~enable', SetBool, self.__set_enable)

        self.__pub_keypoints = rospy.Publisher('~persons', Persons, queue_size=1)
        self.__pub_markers = rospy.Publisher('~markers', MarkerArray, queue_size=10)
        self.__pub_pose = rospy.Publisher('~poses', DetectedPersons, queue_size=10)
        if self.__visualize:
            self.__pub_image = rospy.Publisher('~image', Image, queue_size=10)

        self.__cv_bridge = CvBridge()
        color_sub = message_filters.Subscriber("~color", Image)
        points_sub = message_filters.Subscriber("~points", PointCloud2)
        sub = message_filters.ApproximateTimeSynchronizer([color_sub, points_sub], 10, 0.1)
        sub.registerCallback(self.__callback_image)

    def __set_enable(self, msg):
        if self.__is_active and not msg.data:
            self.__pose_estimator = None
            self.__restart = True
        elif not self.__is_active and msg.data:
            self.__pose_estimator = TfPoseEstimator(
                self.__graph_path,
                target_size=self.__target_size,
                tf_config=self.__tf_config)
        self.__is_active = msg.data
        message = '{} pose estimator.'.format('Enabled' if msg.data else 'Disabled')
        return SetBoolResponse(success=True, message=message)

    def __humans_to_msg(self, humans, points):
        persons = Persons()
        height, width = points.height, points.width

        for human in humans:
            person = Person()

            for k in human.body_parts:
                body_part = human.body_parts[k]

                body_part_msg = BodyPartElm()
                body_part_msg.part_id = body_part.part_idx
                x, y = np.round(body_part.x * width), np.round(body_part.y * height)
                expand = 3
                centers = [[xx, yy] for yy in range(max(int(y - expand), 0), min(int(y + expand), height))
                           for xx in range(max(int(x - expand), 0), min(int(x + expand), width))]
                if k in range(14, 18):
                    centers = [[int(x), int(y)]]
                pts = [p for p in pc2.read_points(points, ('x', 'y', 'z'), uvs=centers, skip_nans=True)]
                if not pts:
                    continue
                pt = np.mean(pts, axis=0)
                body_part_msg.x = pt[0]
                body_part_msg.y = pt[1]
                body_part_msg.z = pt[2]
                body_part_msg.confidence = body_part.score
                person.body_part.append(body_part_msg)
            persons.persons.append(person)

        return persons

    def __callback_image(self, color, points):
        if self.__restart:
            rospy.signal_shutdown('respawn to clean gpu memory')
        if not self.__is_active:
            return
        if rospy.Duration(1. / self.__hz) > rospy.Time.now() - self.__prev_time:
            return

        try:
            cv_image = self.__cv_bridge.imgmsg_to_cv2(color, 'bgr8')
        except CvBridgeError as e:
            rospy.logerr('Converting Image Error. ' + str(e))
            return

        acquired = self.__tf_lock.acquire(False)
        if not acquired:
            return

        self.__prev_time = rospy.Time.now()

        try:
            if self.__pose_estimator is None:
                return
            humans = self.__pose_estimator.inference(
                cv_image, resize_to_default=True, upsample_size=self.__resize_out_ratio)
        finally:
            self.__tf_lock.release()

        msg = self.__humans_to_msg(humans, points)
        msg.image_w = cv_image.shape[1]
        msg.image_h = cv_image.shape[0]
        msg.header = points.header

        self.__pub_keypoints.publish(msg)
        self.__pub_markers.publish(self.__to_markers(msg))
        self.__pub_pose.publish(self.__to_spencer_msg(msg))
        if self.__visualize:
            image = TfPoseEstimator.draw_humans(cv_image, humans, imgcopy=False)
            self.__pub_image.publish(self.__cv_bridge.cv2_to_imgmsg(image, 'bgr8'))

    def __to_markers(self, keypoints):
        markers = MarkerArray()

        links = CocoPairsRender

        markers.markers.append(Marker(header=keypoints.header, action=Marker.DELETEALL))
        for i, p in enumerate(keypoints.persons):
            body_parts = [None] * 18
            for k in p.body_part:
                body_parts[k.part_id] = k

            marker = Marker()
            marker.header = keypoints.header
            marker.ns = 'person_{}'.format(i)
            marker.type = Marker.LINE_LIST
            marker.action = Marker.ADD
            marker.scale.x = 0.05
            marker.lifetime = rospy.Duration(1)
            id = 0
            for ci, link in enumerate(links):
                if body_parts[link[0]] is not None and body_parts[link[1]] is not None:
                    marker.points.append(Point(body_parts[link[0]].x,
                                               body_parts[link[0]].y,
                                               body_parts[link[0]].z))
                    marker.points.append(Point(body_parts[link[1]].x,
                                               body_parts[link[1]].y,
                                               body_parts[link[1]].z))
                    color = CocoColors[ci]
                    marker.id = id
                    id += 1
                    marker.colors.append(ColorRGBA(float(color[0]) / 255,
                                                   float(color[1]) / 255,
                                                   float(color[2]) / 255,
                                                   1.0))
                    marker.colors.append(ColorRGBA(float(color[0]) / 255,
                                                   float(color[1]) / 255,
                                                   float(color[2]) / 255,
                                                   1.0))
            markers.markers.append(marker)

        return markers

    def __to_spencer_msg(self, keypoints):
        persons = DetectedPersons()
        persons.header = keypoints.header
        for p in keypoints.persons:
            for k in p.body_part:
                if k.part_id in [1, 8, 11]:
                    person = DetectedPerson()
                    person.modality = DetectedPerson.MODALITY_GENERIC_RGBD
                    person.pose.pose.position.x = k.x
                    person.pose.pose.position.y = k.y
                    person.pose.pose.position.z = k.z
                    person.confidence = k.confidence
                    person.detection_id = self.__last_detection_id
                    self.__last_detection_id += self.__detection_id_increment
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


if __name__ == '__main__':
    rospy.init_node('tf_pose_estimator')
    node = PoseEstimator()
    rospy.spin()
