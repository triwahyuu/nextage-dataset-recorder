#!/usr/bin/env python

import time
import json
import numpy as np
import cv2
from pathlib import Path

import rospy
import message_filters
import ros_numpy

from cv_bridge import CvBridge
from std_srvs.srv import Trigger, TriggerResponse, TriggerRequest
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs.msg import JointState



class DatasetRecorder:
    def __init__(self):
        rospy.init_node("dataset_recorder", anonymous=True)

        # Parameters
        self.base_save_dir: Path = Path(
            rospy.get_param("~save_dir", "/tmp/dataset_recordings")
        )
        self.image_topic_name: str = rospy.get_param("~image_topic", "/camera/rgb/image_raw").lstrip('/')
        self.pointcloud_topic_name: str = rospy.get_param("~pointcloud_topic", "/camera/depth/points").lstrip('/')
        self.robot_state_topic_name: str = rospy.get_param("~robot_state_topic", "/joint_states").lstrip('/')

        sync_rate: float = rospy.get_param("~sync_rate", 10.0)
        self.sync_period = rospy.Duration(1.0 / sync_rate)

        # Create save directory if it doesn't exist using pathlib
        self.base_save_dir.mkdir(parents=True, exist_ok=True)

        # Initialize variables
        self.is_recording: bool = False
        self.recording_start_time: rospy.Time = None
        self.current_session_dir: Path = None
        self.frame_count: int = 0
        self.last_sync_time = rospy.Time(0)
        self.manifest_data = [] # To store metadata for manifest file
        self.bridge = CvBridge() # Initialize CvBridge

        # --- Directory structure for individual files ---
        self.image_dir: Path = None
        self.pointcloud_dir: Path = None
        self.robot_state_dir: Path = None
        # ---

        # Setup service
        self.record_service = rospy.Service(
            "~record", Trigger, self.handle_start_recording
        )
        self.stop_service = rospy.Service("~stop", Trigger, self.handle_stop_recording)

        # Synchronized subscribers (remain the same)
        self.image_sub = message_filters.Subscriber(self.image_topic_name, Image)
        self.pointcloud_sub = message_filters.Subscriber(
            self.pointcloud_topic_name, PointCloud2
        )
        self.robot_state_sub = message_filters.Subscriber(
            self.robot_state_topic_name, JointState
        )

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.pointcloud_sub, self.robot_state_sub],
            queue_size=10,
            slop=0.1,
        )
        self.ts.registerCallback(self.sync_callback)

        rospy.loginfo(
            "Dataset recorder initialized (Individual File Mode). Waiting for trigger."
        )

    def handle_start_recording(self, req: TriggerRequest) -> TriggerResponse:
        """Service handler to start recording data"""
        if self.is_recording:
            return TriggerResponse(success=False, message="Already recording")

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.current_session_dir = self.base_save_dir / f"session_{timestamp}"

        # Create subdirectories for different data types
        self.image_dir = self.current_session_dir / "rgb"
        self.pointcloud_dir = self.current_session_dir / "pointcloud"
        self.robot_state_dir = self.current_session_dir / "joint_states"

        self.current_session_dir.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(exist_ok=True)
        self.pointcloud_dir.mkdir(exist_ok=True)
        self.robot_state_dir.mkdir(exist_ok=True)

        # Reset manifest data
        self.manifest_data = []

        self.is_recording = True
        self.recording_start_time = rospy.Time.now()
        self.frame_count = 0
        self.last_sync_time = rospy.Time(0) # Reset last sync time

        rospy.loginfo(f"Started recording to {self.current_session_dir}")
        return TriggerResponse(
            success=True, message=f"Started recording to {self.current_session_dir}"
        )

    def handle_stop_recording(self, req: TriggerRequest) -> TriggerResponse:
        """Service handler to stop recording data"""
        if not self.is_recording:
            return TriggerResponse(success=False, message="Not currently recording")

        self.is_recording = False

        # --- Save the manifest file ---
        manifest_path = self.current_session_dir / "manifest.json"
        try:
            with open(manifest_path, 'w') as f:
                json.dump(self.manifest_data, f, indent=4)
            rospy.loginfo(f"Manifest saved to {manifest_path}")
        except Exception as e:
            rospy.logerr(f"Failed to save manifest file: {e}")
        # ---

        duration = (rospy.Time.now() - self.recording_start_time).to_sec()

        rospy.loginfo(
            f"Stopped recording. Recorded {self.frame_count} frames over {duration:.2f} seconds."
        )
        return TriggerResponse(
            success=True,
            message=f"Stopped recording. Recorded {self.frame_count} frames.",
        )

    def sync_callback(
        self, image_msg: Image, pointcloud_msg: PointCloud2, robot_state_msg: JointState
    ) -> None:
        """Callback for synchronized messages"""
        if not self.is_recording:
            return

        current_time = rospy.Time.now() # Use a consistent time for check
        if current_time - self.last_sync_time >= self.sync_period:
            self.last_sync_time = current_time # Use the time we checked

            # Use frame count for consistent naming across modalities
            frame_id_str = f"{self.frame_count:06d}" # e.g., 000000, 000001
            timestamp_sec = image_msg.header.stamp.to_sec() # Use a consistent timestamp (e.g., image)

            # Save individual files
            try:
                img_path_rel = self.save_image(image_msg, frame_id_str)
                pc_path_rel = self.save_pointcloud(pointcloud_msg, frame_id_str)
                js_path_rel = self.save_robot_state(robot_state_msg, frame_id_str)

                # Add entry to manifest
                self.manifest_data.append({
                    "frame_id": self.frame_count,
                    "timestamp": timestamp_sec,
                    "rgb_path": img_path_rel,
                    "pointcloud_path": pc_path_rel,
                    "jointstate_path": js_path_rel
                })

                self.frame_count += 1
                if self.frame_count % 10 == 0: # Log progress occasionally
                     rospy.loginfo(f"Recorded frame {self.frame_count}")

            except Exception as e:
                rospy.logerr(f"Error saving frame {self.frame_count}: {e}")


    def save_image(self, msg: Image, frame_id: str) -> str:
        """Save image message to disk as PNG"""
        try:
            # Convert ROS Image message to OpenCV image (BGR format)
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            filename = f"{frame_id}.png"
            filepath = self.image_dir / filename
            cv2.imwrite(str(filepath), cv_image)
            # Return relative path for manifest
            return f"{self.image_dir.name}/{filename}"
        except Exception as e:
            rospy.logerr(f"Failed to save image frame {frame_id}: {e}")
            raise # Re-raise exception to be caught in sync_callback

    def save_pointcloud(self, msg: PointCloud2, frame_id: str) -> str:
        """Save pointcloud message to disk as NPY"""
        try:
            # Convert PointCloud2 message to NumPy array
            # This creates a structured array, preserving field names
            pc_array = ros_numpy.numpify(msg)

            # Optional: Convert structured array to a simpler N x D array if preferred
            # Example: Get X, Y, Z coordinates
            # points = np.zeros((pc_array.shape[0], 3))
            # points[:, 0] = pc_array['x']
            # points[:, 1] = pc_array['y']
            # points[:, 2] = pc_array['z']
            # If you have RGB:
            # points_rgb = np.zeros((pc_array.shape[0], 6))
            # points_rgb[:, 0] = pc_array['x']
            # ...
            # points_rgb[:, 3] = pc_array['r'] # Or access via 'rgb' field if packed
            # ...
            # pc_to_save = points_rgb # Or just points

            pc_to_save = pc_array # Save the structured array by default

            filename = f"{frame_id}.npy"
            filepath = self.pointcloud_dir / filename
            np.save(str(filepath), pc_to_save)
             # Return relative path for manifest
            return f"{self.pointcloud_dir.name}/{filename}"
        except Exception as e:
            rospy.logerr(f"Failed to save pointcloud frame {frame_id}: {e}")
            raise # Re-raise exception

    def save_robot_state(self, msg: JointState, frame_id: str) -> str:
        """Save robot state message to disk as JSON"""
        try:
            # Create a dictionary from the JointState message
            state_dict = {
                "header_stamp_sec": msg.header.stamp.to_sec(),
                "name": list(msg.name),
                "position": list(msg.position),
                "velocity": list(msg.velocity),
                "effort": list(msg.effort),
            }
            filename = f"{frame_id}.json"
            filepath = self.robot_state_dir / filename
            with open(filepath, 'w') as f:
                json.dump(state_dict, f, indent=4)
             # Return relative path for manifest
            return f"{self.robot_state_dir.name}/{filename}"
        except Exception as e:
            rospy.logerr(f"Failed to save robot state frame {frame_id}: {e}")
            raise # Re-raise exception


if __name__ == "__main__":
    # --- Important Dependencies ---
    # Make sure you have the necessary libraries installed:
    # sudo apt-get install python3-ros-numpy python3-opencv
    # pip install numpy rospkg # rospkg might be needed by ros_numpy indirectly
    # Ensure cv_bridge is available (usually comes with ROS desktop installs)
    # ---

    try:
        recorder = DatasetRecorder()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except ImportError as e:
         # Add specific check for common missing dependencies
        if 'ros_numpy' in str(e):
            print("\n\nERROR: Failed to import 'ros_numpy'.")
            print("Please install it: sudo apt-get install python3-ros-numpy\n\n")
        elif 'cv2' in str(e):
             print("\n\nERROR: Failed to import 'cv2' (OpenCV).")
             print("Please install it: sudo apt-get install python3-opencv\n\n")
        elif 'cv_bridge' in str(e):
             print("\n\nERROR: Failed to import 'cv_bridge'.")
             print("Ensure ROS environment is sourced correctly and cv_bridge is installed.\n\n")
        else:
            print(f"\n\nERROR: Failed to import a required library: {e}\n\n")

