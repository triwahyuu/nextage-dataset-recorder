import time
import json
import pickle
import rospy
import message_filters

from datetime import datetime
from pathlib import Path
from std_srvs.srv import Trigger, TriggerResponse, TriggerRequest

from dataset_recorder.camera import DualKinectRecorder
from dataset_recorder.robot import RobotStateRecorder
from dataset_recorder.utils import map_subinfo_to_idx


class DatasetRecorder:
    def __init__(self):
        rospy.init_node("dataset_recorder", anonymous=True)

        # --- Core Parameters ---
        self.base_save_dir: Path = Path(
            rospy.get_param("~save_dir", "/tmp/dataset_recordings")
        )
        self.sync_rate: float = rospy.get_param("~sync_rate", 10.0)
        self.sync_period = rospy.Duration(1.0 / self.sync_rate)
        self.queue_size = rospy.get_param("~queue_size", 10)
        self.slop = rospy.get_param("~slop", 0.1)

        # Initialize variables
        self.is_recording: bool = False
        self.recording_start_time: rospy.Time = None
        self.clip_dir: Path = None
        self.frame_count: int = 0
        self.last_sync_time = rospy.Time(0)
        self.frame_info = []
        self.attribute_data = {}

        # Setup service
        self.record_service = rospy.Service(
            "~start", Trigger, self.handle_start_recording
        )
        self.stop_service = rospy.Service("~stop", Trigger, self.handle_stop_recording)

        # --- Setup Subscribers ---
        self.subscribers = []
        self.subscriber_info = []

        self.camera = DualKinectRecorder(self.base_save_dir)
        self.robot = RobotStateRecorder()

        # --- Setup Synchronizer ---
        self.camera.setup(self.subscribers, self.subscriber_info)
        self.robot.setup(self.subscribers, self.subscriber_info)

        self.msg_idx_map = map_subinfo_to_idx(self.subscriber_info)
        self.camera.set_msg_idx_map(self.msg_idx_map)
        self.robot.set_msg_idx_map(self.msg_idx_map)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            self.subscribers,
            queue_size=self.queue_size,
            slop=self.slop,
        )
        self.ts.registerCallback(self.sync_callback)
        # ---

        rospy.loginfo("Dataset recorder initialized.")
        rospy.loginfo("Waiting for trigger to start recording.")

    def handle_start_recording(self, req: TriggerRequest) -> TriggerResponse:
        """Service handler to start recording data"""
        if self.is_recording:
            return TriggerResponse(success=False, message="Already recording")

        timestamp = rospy.Time.now().to_nsec() // 1000  # microseconds
        timestamp_str = f"{timestamp}"
        self.clip_dir: Path = self.base_save_dir / timestamp_str
        self.msgs_dir = self.clip_dir / "msgs"
        self.msgs_dir.mkdir(parents=True, exist_ok=True)

        self.camera.set_clip_name(timestamp_str)

        # Reset manifest data and get attributes
        self.frame_info = []
        self.attribute_data = {
            "camera": self.camera.get_attributes(),
            "robot": self.robot.get_attributes(),
            "message_idx_map": self.msg_idx_map,
        }

        self.is_recording = True
        self.recording_start_time = rospy.Time.now()
        self.frame_count = 0
        self.last_sync_time = rospy.Time(0)  # Reset last sync time

        rospy.loginfo(f"Started recording to {self.clip_dir}")
        return TriggerResponse(
            success=True, message=f"Started recording to {self.clip_dir}"
        )

    def handle_stop_recording(self, req: TriggerRequest) -> TriggerResponse:
        """Service handler to stop recording data"""
        if not self.is_recording:
            return TriggerResponse(success=False, message="Not currently recording")

        self.is_recording = False

        # --- Save the manifest file ---
        attributes_path = self.clip_dir / "attributes.json"
        frame_info_file = "frame_info.json"
        frame_info_path = self.clip_dir / frame_info_file

        time_now = rospy.Time.now()
        start_time_str = datetime.fromtimestamp(
            self.recording_start_time.to_sec()
        ).strftime("%Y%m%d_%H%M%S")
        end_time_str = datetime.fromtimestamp(time_now.to_sec()).strftime(
            "%Y%m%d_%H%M%S"
        )
        duration_sec = (
            (time_now - self.recording_start_time).to_sec()
            if self.recording_start_time
            else 0
        )
        try:
            final_metadata = {
                "total_frames": self.frame_count,
                "duration_sec": duration_sec,
                "start_time": start_time_str,
                "end_time": end_time_str,
            }
            attributes_data = {
                "metadata": final_metadata,
                "attributes": self.attribute_data,
                "task_info": {},
                "frame_info_path": frame_info_file,
            }
            with open(attributes_path, "w") as f:
                json.dump(attributes_data, f)
            with open(frame_info_path, "w") as f:
                json.dump(self.frame_info, f)
            rospy.loginfo(f"Manifest saved to {attributes_path}")
        except Exception as e:
            rospy.logerr(f"Failed to save manifest file: {e}")
        # ---

        rospy.loginfo(
            f"Stopped recording. Recorded {self.frame_count} frames over {duration_sec:.2f} seconds."
        )
        # Reset state
        self.clip_dir = None
        self.recording_start_time = None
        self.frame_info = []

        return TriggerResponse(
            success=True,
            message=f"Stopped recording. Recorded {self.frame_count} frames.",
        )

    def sync_callback(self, *msgs) -> None:
        """Callback for synchronized messages"""
        if not self.is_recording:
            return

        current_time = rospy.Time.now()
        interval = current_time - self.last_sync_time
        # if interval < self.sync_period:
        #     return

        self.last_sync_time = current_time

        frame_id_str = f"{self.frame_count:06d}"  # e.g., 000000, 000001
        frame_info = {
            "frame_id": frame_id_str,
            "timestamp": current_time.to_sec(),
        }
        try:
            # frame_info["rgb_path"] = self.camera.save_image(msgs, frame_id_str)
            # frame_info["depth_path"] = self.camera.save_depth(msgs, frame_id_str)
            # frame_info["pcd_path"] = self.camera.save_pointcloud(msgs, frame_id_str)
            frame_info["robot_states"] = self.robot.get_robot_state(msgs, frame_id_str)
            msgs_path = self.msgs_dir / f"{frame_id_str}.pkl"
            with open(msgs_path, "wb") as f:
                pickle.dump(msgs, f)

            frame_info["messages_path"] = str(msgs_path.relative_to(self.clip_dir))
            self.frame_info.append(frame_info)

            self.frame_count += 1
            rospy.loginfo(f"Recorded frame {self.frame_count}  {interval.to_sec()}")

        except Exception as e:
            rospy.logerr(
                f"Error saving data for frame {self.frame_count}: {e}",
                exc_info=True,
            )
