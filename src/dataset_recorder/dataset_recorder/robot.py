import json
import numpy as np

from typing import List
from pathlib import Path

import rospy
import message_filters
from sensor_msgs.msg import JointState


class ArmRecorder:
    """
    Handles recording RGB images and PointCloud2 data from a Kinect-like
    sensor running in ROS to a specified directory.
    """
    def __init__(self, output_dir, which_arm):
        """
        Initializes the recorder.

        Args:
            output_dir (str): The base directory where data will be saved.
            which_arm (str): Which arm to record, either 'left' or 'right'.
        """
        rospy.loginfo(f"Initializing Kinect Recorder...")

        self.which_arm = which_arm
        self.output_dir = Path(output_dir).joinpath(which_arm).resolve()

        robot_state_topic = rospy.get_param("~robot_state_topic", "/joint_states").lstrip('/')
        self.robot_state_topic_name: str = robot_state_topic

        self.robot_state_dir = self.output_dir / "robot_state"
        self.robot_state_dir.mkdir(parents=True, exist_ok=True)

        self.robot_state_sub = None

    def setup(self, subscribers: list, subscriber_info: list):
        """Sets up the ROS subscribers."""
        self.robot_state_sub = message_filters.Subscriber(self.robot_state_topic_name, JointState)
        self.state_info_name = f"{self.which_arm}_arm/state"
        subscribers.append(self.robot_state_sub)
        subscriber_info.append({"name": self.state_info_name, "type": JointState})

    def save_robot_state(self, msgs: list, frame_id: str) -> str:
        """Save image message to disk as PNG"""
        filename = f"{frame_id}.json"
        filepath = self.robot_state_dir / filename

        try:
            msg: JointState = msgs[self.state_info_name]
            state_dict = {
                "header_stamp_sec": msg.header.stamp.to_sec(),
                "name": list(msg.name),
                "position": list(msg.position) if msg.position is not None else [],
                "velocity": list(msg.velocity) if msg.velocity is not None else [],
                "effort": list(msg.effort) if msg.effort is not None else [],
            }
            with open(filepath, 'w') as f:
                json.dump(state_dict, f, indent=4)
            return filepath
        except Exception as e:
            rospy.logerr(f"Failed to save robot state frame {frame_id}: {e}")
            raise

class DualArmRecorder:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir).resolve()
        self.left_recorder = ArmRecorder(output_dir, "left")
        self.right_recorder = ArmRecorder(output_dir, "right")

    def setup(self, subscribers: list, subscriber_info: list):
        self.left_recorder.setup(subscribers, subscriber_info)
        self.right_recorder.setup(subscribers, subscriber_info)

    def save_robot_state(self, msgs: list, frame_id: str) -> List[str]:
        left_filepath = self.left_recorder.save_robot_state(msgs, frame_id)
        right_filepath = self.right_recorder.save_robot_state(msgs, frame_id)
        return [left_filepath, right_filepath]