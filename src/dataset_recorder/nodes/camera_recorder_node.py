#!/usr/bin/env python3

import rospy
import time
import pickle

from pathlib import Path
from std_srvs.srv import Trigger, TriggerResponse, TriggerRequest
from dataset_recorder.camera import DualKinectRecorder
from dataset_recorder.utils import map_subinfo_to_idx


class CameraRecorder:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

        self.frame_idx = 0
        self.clip_name = None
        self.clip_dir = None

        self.recorder = DualKinectRecorder(output_dir)
        sub, sub_info = [], []
        self.recorder.setup(sub, sub_info)
        msg_idx_map = map_subinfo_to_idx(sub_info)
        self.recorder.set_msg_idx_map(msg_idx_map)

        self.record_service = rospy.Service("~start", Trigger, self.start_record)
        self.stop_service = rospy.Service("~stop", Trigger, self.stop_record)
        self.is_recording = False

    def start_record(self, req: TriggerRequest) -> TriggerResponse:
        if self.is_recording:
            return TriggerResponse(success=False, message="Already recording")

        self.is_recording = True
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        self.clip_name = timestamp
        self.clip_dir = self.output_dir / self.clip_name
        self.msgs_dir = self.clip_dir / "msgs"
        self.msgs_dir.mkdir(parents=True, exist_ok=True)
        rospy.loginfo(f"Saving to: {self.clip_dir}")

        self.recorder.reset_output_dir(self.clip_dir)
        return TriggerResponse(
            success=True, message=f"Started recording to {self.clip_dir}"
        )

    def stop_record(self, req: TriggerRequest) -> TriggerResponse:
        if not self.is_recording:
            return TriggerResponse(success=False, message="Not currently recording")

        self.is_recording = False

        return TriggerResponse(
            success=True,
            message=f"Stopped recording. Recorded {self.frame_idx} frames.",
        )

    def rate_callback(self, *msgs):
        if not self.is_recording:
            return

        start_ts = time.perf_counter()
        output_file = self.msgs_dir / f"{self.frame_idx:06d}.pkl"
        with output_file.open("wb") as f:
            pickle.dump(msgs, f)
        rospy.loginfo(f"{self.frame_idx}  {(time.perf_counter() - start_ts)*1000} ms")
        self.frame_idx += 1


if __name__ == "__main__":
    try:
        rospy.init_node("camera_recorder", anonymous=True)
        output_dir = Path(rospy.get_param("~save_dir", "/tmp/dataset_recordings"))

        recorder = CameraRecorder(output_dir)
        recorder.start_record()

        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as main_e:
        rospy.logfatal(f"Unhandled exception in main: {main_e}", exc_info=True)
