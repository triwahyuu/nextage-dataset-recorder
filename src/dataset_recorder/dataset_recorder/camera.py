import cv2
import numpy as np

from typing import List
from pathlib import Path

import rospy
import message_filters
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge


class KinectRecorder:
    """
    Handles recording RGB images and PointCloud2 data from a Kinect-like
    sensor running in ROS to a specified directory.
    """
    def __init__(self, output_dir, node_ns):
        """
        Initializes the recorder.

        Args:
            output_dir (str): The base directory where data will be saved.
            node_ns (str): The ROS topic namespace for the node.
        """
        rospy.loginfo(f"Initializing Kinect Recorder...")

        self.node_ns = f"{node_ns}/"
        self.output_dir = Path(output_dir).joinpath(node_ns).resolve()

        img_topic = rospy.get_param("~image_topic", "/rgb/image_raw").lstrip('/')
        self.image_topic_name = f"/{self.node_ns}{img_topic}"

        pc_topic = rospy.get_param("~pointcloud_topic", "/depth_registered/points").lstrip('/')
        self.pc_topic_name = f"/{self.node_ns}{pc_topic}"

        self.image_dir = self.output_dir / "rgb"
        self.pc_dir = self.output_dir / "point_cloud"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.pc_dir.mkdir(parents=True, exist_ok=True)

        self.bridge = CvBridge()
        self.image_sub = None
        self.pc_sub = None

        self.rgb_count = 0
        self.pc_count = 0

    def setup(self, subscribers: list, subscriber_info: list):
        """Sets up the ROS subscribers."""
        self.image_sub = message_filters.Subscriber(self.image_topic_name, Image)
        self.img_info_name = f"{self.node_ns}image"
        subscribers.append(self.image_sub)
        subscriber_info.append({"name": self.img_info_name, "type": Image})

        self.pc_sub = message_filters.Subscriber(self.pc_topic_name, PointCloud2)
        self.pc_info_name = f"{self.node_ns}pointcloud"
        subscribers.append(self.pc_sub)
        subscriber_info.append({"name": self.pc_info_name, "type": PointCloud2})

    def save_image(self, msgs: list, frame_id: str) -> str:
        """Save image message to disk as PNG"""
        filename = f"{frame_id}.png"
        filepath = self.image_dir / filename

        try:
            msg: Image = msgs[self.img_info_name]
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            cv2.imwrite(str(filepath), cv_image)
            return filepath
        except Exception as e:
            rospy.logerr(f"Failed to save image frame {frame_id}: {e}")
            raise

    def save_pointcloud(self, msgs: dict, frame_id: str) -> str:
        """Save pointcloud message to disk as NPY (XYZRGB)"""
        filename = f"{frame_id}.npy"
        filepath = self.pc_dir / filename

        try:
            # Convert PointCloud2 to numpy array
            msg: PointCloud2 = msgs[self.pc_info_name]
            pc_data = pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z", "rgb"))
            pc_array = np.array(list(pc_data))

            np.save(filepath, pc_array)
            return filepath
        except Exception as e:
            rospy.logerr(f"Failed to save pointcloud frame {frame_id}: {e}")
            raise


class DualKinectRecorder:
    def __init__(self, output_dir):
        self.base_dir = Path(output_dir)

        self.left_recorder = KinectRecorder(output_dir, "kinect_left")
        self.right_recorder = KinectRecorder(output_dir, "kinect_right")

    def setup(self, subscribers: list, subscriber_info: list):
        self.left_recorder.setup(subscribers, subscriber_info)
        self.right_recorder.setup(subscribers, subscriber_info)

    def save_image(self, msgs: dict, frame_id: str) -> List[str]:
        left_path = self.left_recorder.save_image(msgs, frame_id)
        right_path = self.right_recorder.save_image(msgs, frame_id)
        return [left_path, right_path]

    def save_pointcloud(self, msgs: dict, frame_id: str) -> List[str]:
        left_path = self.left_recorder.save_pointcloud(msgs, frame_id)
        right_path = self.right_recorder.save_pointcloud(msgs, frame_id)
        return [left_path, right_path]


if __name__ == "__main__":
    rospy.init_node("kinect_recorder", anonymous=True)
    output_dir = rospy.get_param("~output_dir", "/workspaces/dataset_recorder/playground/kinect_recordings")
    recorder = KinectRecorder(output_dir, "kinect")

    def sync_callback(*msgs):
        try:
            recorder.save_image(msgs, "test")
            recorder.save_pointcloud(msgs, "test")
        except Exception as e:
            rospy.logerr(f"Error saving data: {e}")

    subs, sub_info = [], []
    recorder.setup(subs, sub_info)
    ts = message_filters.ApproximateTimeSynchronizer(
        subs, queue_size=10, slop=0.1
    )
    ts.registerCallback(sync_callback)
    rospy.spin()

