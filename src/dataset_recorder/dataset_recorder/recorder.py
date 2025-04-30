import time
import json
from pathlib import Path

import rospy
import message_filters

from std_srvs.srv import Trigger, TriggerResponse, TriggerRequest
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped

from camera import DualKinectRecorder
from robot import ArmRecorder
from utils import map_subinfo_to_idx


class DatasetRecorder:
    def __init__(self):
        rospy.init_node("dataset_recorder", anonymous=True)

        # --- Core Parameters ---
        self.base_save_dir: Path = Path(
            rospy.get_param("~save_dir", "/tmp/dataset_recordings")
        )
        sync_rate: float = rospy.get_param("~sync_rate", 10.0)
        self.sync_period = rospy.Duration(1.0 / sync_rate)
        self.queue_size = rospy.get_param("~queue_size", 10)
        self.slop = rospy.get_param("~slop", 0.1)

        # Initialize variables
        self.is_recording: bool = False
        self.recording_start_time: rospy.Time = None
        self.current_session_dir: Path = None
        self.frame_count: int = 0
        self.last_sync_time = rospy.Time(0)
        self.manifest_data = [] # To store metadata for manifest file

        # --- Directory structure paths ---
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.current_session_dir: Path = self.base_save_dir / f"{timestamp}"
        self.robot_state_dir: Path = self.current_session_dir / "joint_states"
        self.data_dir: Path = self.current_session_dir / "data"

        # Setup service
        self.record_service = rospy.Service(
            "~record", Trigger, self.handle_start_recording
        )
        self.stop_service = rospy.Service("~stop", Trigger, self.handle_stop_recording)

        # --- Setup Subscribers ---
        self.subscribers = []
        self.subscriber_info = [] # Keep track of topic name and type for callback mapping

        # Core Subscribers
        self.robot_state_sub = message_filters.Subscriber(self.robot_state_topic_name, JointState)
        self.subscribers.append(self.robot_state_sub)
        self.subscriber_info.append({"name": "robot_state", "type": JointState})

        # cameras
        self.camera = DualKinectRecorder(self.base_save_dir)

        # robot
        # TODO: FINISH
        # self.robot = ArmRecorder(self.base_save_dir, "left")

        # --- Setup Synchronizer ---
        self.camera.setup(self.subscribers, self.subscriber_info)
        # self.robot.setup(self.subscribers, self.subscriber_info)

        self.msg_idx_map = map_subinfo_to_idx(self.subscriber_info)
        self.camera.set_msg_idx_map(self.msg_idx_map)
        # self.robot.set_msg_idx_map(self.msg_idx_map)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            self.subscribers,
            queue_size=self.queue_size,
            slop=self.slop,
        )
        self.ts.registerCallback(self.sync_callback)
        # ---

        rospy.loginfo(
            f"Dataset recorder initialized. Saving to '{self.base_save_dir}'. Waiting for trigger."
        )

    def handle_start_recording(self, req: TriggerRequest) -> TriggerResponse:
        """Service handler to start recording data"""
        if self.is_recording:
            return TriggerResponse(success=False, message="Already recording")

        self.robot_state_dir.mkdir(exist_ok=True)
        self.current_session_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)

        # Reset manifest data and save config
        self.manifest_data = []
        self.save_config() # Save configuration for this session

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
            # Add final metadata to manifest (optional)
            final_metadata = {
                "total_frames": self.frame_count,
                "recording_duration_sec": (rospy.Time.now() - self.recording_start_time).to_sec() if self.recording_start_time else 0,
                "end_time": time.strftime("%Y%m%d_%H%M%S")
            }
            # Save manifest as a dictionary with metadata and frame list
            manifest_content = {
                "metadata": final_metadata,
                "frames": self.manifest_data
            }
            with open(manifest_path, 'w') as f:
                json.dump(manifest_content, f, indent=4)
            rospy.loginfo(f"Manifest saved to {manifest_path}")
        except Exception as e:
            rospy.logerr(f"Failed to save manifest file: {e}")
        # ---

        duration = (rospy.Time.now() - self.recording_start_time).to_sec() if self.recording_start_time else 0

        rospy.loginfo(
            f"Stopped recording. Recorded {self.frame_count} frames over {duration:.2f} seconds."
        )
        # Reset state
        self.current_session_dir = None
        self.recording_start_time = None
        self.manifest_data = []

        return TriggerResponse(
            success=True,
            message=f"Stopped recording. Recorded {self.frame_count} frames.",
        )

    def save_config(self):
        """Saves the node's configuration to a JSON file in the session directory."""
        if not self.current_session_dir:
            return
        config_path = self.current_session_dir / "config.json"
        config_data = {
            "save_dir": str(self.base_save_dir),
            "sync_rate": 1.0 / self.sync_period.to_sec() if self.sync_period.to_sec() > 0 else float('inf'),
            "queue_size": self.queue_size,
            "slop": self.slop
        }
        try:
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=4)
            rospy.loginfo(f"Configuration saved to {config_path}")
        except Exception as e:
            rospy.logerr(f"Failed to save config file: {e}")


    def sync_callback(self, *msgs) -> None:
        """Callback for synchronized messages"""
        if not self.is_recording:
            return

        current_time = rospy.Time.now() # Use a consistent time for check
        if current_time - self.last_sync_time >= self.sync_period:
            self.last_sync_time = current_time # Use the time we checked

            # Use frame count for consistent naming across modalities
            frame_id_str = f"{self.frame_count:06d}" # e.g., 000000, 000001

            # Use a consistent timestamp (e.g., from image or robot state header)
            # Using robot_state as it often reflects the control loop time better
            # timestamp_sec = received_data["robot_state"].header.stamp.to_sec()

            frame_info = {
                "frame_id": self.frame_count,
                "timestamp": rospy.Time.now().to_sec(),
            }
            # Save individual files
            try:
                # Core data
                frame_info["rgb_path"] = self.camera.save_image(msgs, frame_id_str)
                frame_info["depth_path"] = self.camera.save_depth(msgs, frame_id_str)
                frame_info["pcd_path"] = self.camera.save_pointcloud(msgs, frame_id_str)
                # frame_info["jointstate_path"] = self.robot.save_robot_state(msgs, frame_id_str)

                self.manifest_data.append(frame_info)

                self.frame_count += 1
                if self.frame_count % 20 == 0:
                    rospy.loginfo(f"Recorded frame {self.frame_count}")

            except Exception as e:
                rospy.logerr(f"Error saving data for frame {self.frame_count}: {e}", exc_info=True)

    def convert_ee_pose(self, msg: PoseStamped):
        pose_dict = {
                "frame_id": msg.header.frame_id,
                "position": {
                    "x": msg.pose.position.x,
                    "y": msg.pose.position.y,
                    "z": msg.pose.position.z,
                },
                "orientation": {
                    "x": msg.pose.orientation.x,
                    "y": msg.pose.orientation.y,
                    "z": msg.pose.orientation.z,
                    "w": msg.pose.orientation.w,
                },
            }
        return pose_dict

    def save_data(self, data_dict, frame_id: str) -> str:
        """Save data dictionary to disk as JSON"""
        try:
            filename = f"{frame_id}.json"
            filepath = self.data_dir / filename
            with open(filepath, 'w') as f:
                json.dump(data_dict, f, indent=4)
            return f"{self.data_dir.name}/{filename}"
        except Exception as e:
            rospy.logerr(f"Failed to save data frame {frame_id}: {e}")
            raise
