import cv2
import numpy as np

from typing import List
from pathlib import Path

import rospy
import message_filters
import sensor_msgs.point_cloud2 as pc2
import tf2_ros
import tf2_sensor_msgs
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
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
        rospy.loginfo(f"Initializing Kinect Recorder '{node_ns}'...")

        self.node_ns = f"{node_ns}"
        self.output_dir = Path(output_dir).joinpath(node_ns).resolve()

        img_topic = rospy.get_param("~image_topic", "/rgb/image_raw").lstrip('/')
        self.image_topic_name = f"/{self.node_ns}/{img_topic}"
        self.img_info_name = f"{self.node_ns}/image"

        depth_topic = rospy.get_param("~depth_topic", "/depth/image_raw").lstrip('/')
        self.depth_topic_name = f"/{self.node_ns}/{depth_topic}"
        self.depth_info_name = f"{self.node_ns}/depth"

        depth_reg_topic = rospy.get_param("~depth_reg_topic", "/depth_to_rgb/hw_registered/image_rect_raw").lstrip('/')
        self.depth_reg_topic_name = f"/{self.node_ns}/{depth_reg_topic}"
        self.depth_reg_info_name = f"{self.node_ns}/depth_reg"

        pc_topic = rospy.get_param("~pointcloud_topic", "/points2").lstrip('/')
        self.pc_topic_name = f"/{self.node_ns}/{pc_topic}"
        self.pc_info_name = f"{self.node_ns}/pointcloud"

        self.tf_name = f"{node_ns}_camera_base"
        self.base_frame = "WAIST"
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.camera_pose_tf = None

        self.image_dir = self.output_dir / "rgb"
        self.depth_dir = self.output_dir / "depth"
        self.pc_dir = self.output_dir / "point_cloud"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.depth_dir.mkdir(parents=True, exist_ok=True)
        self.pc_dir.mkdir(parents=True, exist_ok=True)

        self.bridge = CvBridge()
        self.image_sub = None
        self.pc_sub = None
        self.msg_idx_map = {}

        self.rgb_count = 0
        self.pc_count = 0

        rgb_caminfo_topic = rospy.get_param("~rgb_caminfo_topic", "/rgb/camera_info").lstrip('/')
        self.rgb_caminfo_topic = f"/{self.node_ns}/{rgb_caminfo_topic}"
        self.rgb_caminfo = None

        depth_caminfo_topic = rospy.get_param("~depth_caminfo_topic", "/depth/camera_info").lstrip('/')
        self.depth_caminfo_topic = f"/{self.node_ns}/{depth_caminfo_topic}"
        self.depth_caminfo = None

    def setup(self, subscribers: list, subscriber_info: list):
        """Sets up the ROS subscribers."""
        self.image_sub = message_filters.Subscriber(self.image_topic_name, Image)
        subscribers.append(self.image_sub)
        subscriber_info.append({"name": self.img_info_name, "type": Image})

        self.depth_sub = message_filters.Subscriber(self.depth_topic_name, Image)
        subscribers.append(self.depth_sub)
        subscriber_info.append({"name": self.depth_info_name, "type": Image})

        self.depth_reg_sub = message_filters.Subscriber(self.depth_reg_topic_name, Image)
        subscribers.append(self.depth_reg_sub)
        subscriber_info.append({"name": self.depth_reg_info_name, "type": Image})

        self.pc_sub = message_filters.Subscriber(self.pc_topic_name, PointCloud2)
        subscribers.append(self.pc_sub)
        subscriber_info.append({"name": self.pc_info_name, "type": PointCloud2})
        self.pcd_frameid = f"{self.node_ns}_rgb_camera_link"
        self.pcd2base = None

        rgb_caminfo = rospy.wait_for_message(self.rgb_caminfo_topic, CameraInfo, timeout=1)
        self.rgb_caminfo = self.caminfo_to_dict(rgb_caminfo)

        depth_caminfo = rospy.wait_for_message(self.depth_caminfo_topic, CameraInfo, timeout=1)
        self.depth_caminfo = self.caminfo_to_dict(depth_caminfo)

        self.camera_pose_tf = self.get_camera_pose()
        self.pcd2base = self.get_pose_tf(self.pcd_frameid, self.base_frame)

    def set_msg_idx_map(self, idx_map: dict):
        self.msg_idx_map = idx_map

    def get_attributes(self):
        if self.rgb_caminfo is None or self.depth_caminfo is None:
            raise RuntimeError("Camera info not available yet. Call setup() first.")

        return {
            "cam_pose": self.camera_pose_tf,
            "cam_info": {
                "rgb": self.rgb_caminfo,
                "depth": self.depth_caminfo,
            },
            "pcd_frame": self.base_frame
        }

    def get_pose_tf(self, tf_name: str, base_frame: str) -> tf2_ros.TransformStamped:
        try:
            # Get the latest transform from base_frame to the camera's frame
            transform_stamped: tf2_ros.TransformStamped = self.tf_buffer.lookup_transform(
                base_frame, tf_name, rospy.Time(0), rospy.Duration(0.1) # Short timeout
            )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"[KinectRecorder.{self.node_ns}] Could not get transform from '{base_frame}' to '{tf_name}': {e}")
            return None
        return transform_stamped

    def get_pose_tf_dict(self, tf_name: str, base_frame: str):
        transform_stamped = self.get_pose_tf(tf_name, base_frame)
        if transform_stamped is None:
            return None

        return {
                "base_frame_id": transform_stamped.header.frame_id,
                "frame_id": transform_stamped.child_frame_id,
                "translation": {"x": transform_stamped.transform.translation.x, "y": transform_stamped.transform.translation.y, "z": transform_stamped.transform.translation.z},
                "rotation": {"x": transform_stamped.transform.rotation.x, "y": transform_stamped.transform.rotation.y, "z": transform_stamped.transform.rotation.z, "w": transform_stamped.transform.rotation.w}
            }

    def get_camera_pose(self):
        return {
            "camera2world": self.get_pose_tf_dict(self.tf_name, self.base_frame),
            "depth2camera": self.get_pose_tf_dict(f"{self.node_ns}_depth_camera_link", self.tf_name),
            "rgb2camera": self.get_pose_tf_dict(f"{self.node_ns}_rgb_camera_link", self.tf_name),
        }

    @staticmethod
    def caminfo_to_dict(msg: CameraInfo):
        return {
            "height": msg.height,
            "width": msg.width,
            "distortion_model": msg.distortion_model,
            "D": list(msg.D),  # Distortion coefficients
            "K": list(msg.K),  # Intrinsic camera matrix
            "R": list(msg.R),  # Rectification matrix
            "P": list(msg.P),  # Projection/camera matrix
            "binning_x": msg.binning_x,
            "binning_y": msg.binning_y,
            "roi": {
                "x_offset": msg.roi.x_offset,
                "y_offset": msg.roi.y_offset,
                "height": msg.roi.height,
                "width": msg.roi.width,
                "do_rectify": msg.roi.do_rectify,
            },
        }

    def save_image(self, msgs: list, frame_id: str) -> str:
        """Save image message to disk as PNG"""
        filename = f"{frame_id}.png"
        filepath = self.image_dir / filename

        try:
            msg: Image = msgs[self.msg_idx_map[self.img_info_name]]
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            cv2.imwrite(str(filepath), cv_image)
            return filepath
        except Exception as e:
            rospy.logerr(f"Failed to save image frame {frame_id}: {e}")
            raise

    def save_depth(self, msgs: list, frame_id: str) -> str:
        """Save depth message to disk as PNG"""
        filename = f"{frame_id}.png"
        filepath = self.depth_dir / filename
        filename_reg = f"{frame_id}_aligned.png"
        filepath_reg = self.depth_dir / filename_reg

        try:
            msg: Image = msgs[self.msg_idx_map[self.depth_info_name]]
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            cv2.imwrite(str(filepath), cv_image)

            msg: Image = msgs[self.msg_idx_map[self.depth_reg_info_name]]
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            cv2.imwrite(str(filepath_reg), cv_image)
            return filepath_reg
        except Exception as e:
            rospy.logerr(f"Failed to save depth frame {frame_id}: {e}")
            raise

    def save_pointcloud(self, msgs: list, frame_id: str) -> str:
        """Save pointcloud message to disk as NPY (XYZRGB)"""
        filename = f"{frame_id}.npy"
        filepath = self.pc_dir / filename

        try:
            # Convert PointCloud2 to numpy array
            msg: PointCloud2 = msgs[self.msg_idx_map[self.pc_info_name]]
            if self.pcd_frameid != msg.header.frame_id:
                self.pcd_frameid = msg.header.frame_id
                self.pcd2base = self.get_pose_tf(self.pcd_frameid, self.base_frame)
            transformed_msg = tf2_sensor_msgs.do_transform_cloud(msg, self.pcd2base)
            pc_data = pc2.read_points(transformed_msg, skip_nans=True, field_names=("x", "y", "z", "rgb"))
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

    def set_msg_idx_map(self, idx_map: dict):
        self.left_recorder.set_msg_idx_map(idx_map)
        self.right_recorder.set_msg_idx_map(idx_map)

    def save_image(self, msgs: list, frame_id: str) -> List[str]:
        left_path = self.left_recorder.save_image(msgs, frame_id)
        right_path = self.right_recorder.save_image(msgs, frame_id)
        return {"l": left_path, "r": right_path}

    def save_depth(self, msgs: list, frame_id: str) -> List[str]:
        left_path = self.left_recorder.save_depth(msgs, frame_id)
        right_path = self.right_recorder.save_depth(msgs, frame_id)
        return {"l": left_path, "r": right_path}

    def save_pointcloud(self, msgs: list, frame_id: str) -> List[str]:
        left_path = self.left_recorder.save_pointcloud(msgs, frame_id)
        right_path = self.right_recorder.save_pointcloud(msgs, frame_id)
        return {"l": left_path, "r": right_path}

    def get_attributes(self):
        return {
            "left": self.left_recorder.get_attributes(),
            "right": self.right_recorder.get_attributes()
        }


if __name__ == "__main__":
    from dataset_recorder.utils import map_subinfo_to_idx

    rospy.init_node("kinect_recorder", anonymous=True)
    output_dir = rospy.get_param("~output_dir", "/workspaces/dataset_recorder/playground/kinect_recordings")
    recorder = KinectRecorder(output_dir, "kinect_left")

    subs, sub_info = [], []
    recorder.setup(subs, sub_info)
    recorder.set_msg_idx_map(map_subinfo_to_idx(sub_info))

    frame_num = 0

    def sync_callback(*msgs):
        global frame_num
        try:
            frame_id = f"{frame_num:06d}"
            img_paths = recorder.save_image(msgs, frame_id)
            depth_paths = recorder.save_depth(msgs, frame_id)
            pcd_paths = recorder.save_pointcloud(msgs, frame_id)
            frame_num += 1
        except Exception as e:
            rospy.logerr(f"Error saving data: {e}")

    ts = message_filters.ApproximateTimeSynchronizer(
        subs, queue_size=10, slop=0.1
    )
    ts.registerCallback(sync_callback)
    rospy.spin()
