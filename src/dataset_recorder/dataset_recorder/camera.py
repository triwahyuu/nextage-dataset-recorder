import cv2
import threading
import numpy as np
from pathlib import Path

import rospy
import message_filters
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge, CvBridgeError


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

        self.node_ns = node_ns
        self.output_dir = Path(output_dir).joinpath(node_ns).resolve()

        img_topic = rospy.get_param("~image_topic", "/rgb/image_raw").lstrip('/')
        self.image_topic_name = f"/{node_ns}/{img_topic}"

        pc_topic = rospy.get_param("~pointcloud_topic", "/depth_registered/points").lstrip('/')
        self.pc_topic_name = f"/{node_ns}/{pc_topic}"

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
        subscribers.append(self.image_sub)
        subscriber_info.append({"name": f"{self.node_ns}/image", "type": Image})

        self.pc_sub = message_filters.Subscriber(self.pc_topic_name, PointCloud2)
        subscribers.append(self.pc_sub)
        subscriber_info.append({"name": f"{self.node_ns}/pointcloud", "type": PointCloud2})

    def save_image(self, msg: Image) -> str:
        """Save image message to disk as PNG"""
        # Use message timestamp for filename
        timestamp = msg.header.stamp.to_sec()
        filename = f"{timestamp:.6f}.png"
        filepath = self.image_dir / filename

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            cv2.imwrite(str(filepath), cv_image)
            return f"{self.image_dir.name}/{filename}" # Relative path
        except Exception as e:
            rospy.logerr(f"Failed to save image frame {filepath}: {e}")
            raise

    def save_pointcloud(self, msg: PointCloud2) -> str:
        """Save pointcloud message to disk as NPY (XYZRGB)"""
        # Use message timestamp for filename
        timestamp = msg.header.stamp.to_sec()
        filename = f"{timestamp:.6f}.npy"
        filepath = self.pc_dir / filename

        try:
            # Convert PointCloud2 to numpy array
            pc_data = pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z", "rgb"))
            pc_array = np.array(list(pc_data))

            np.save(filepath, pc_array)
            return f"{self.pc_dir.name}/{filename}" # Relative path
        except Exception as e:
            rospy.logerr(f"Failed to save pointcloud frame {filepath}: {e}")
            raise
