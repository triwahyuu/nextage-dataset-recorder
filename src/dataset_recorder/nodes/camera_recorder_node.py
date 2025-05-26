#!/usr/bin/env python3

import rospy
import time
import pickle

from pathlib import Path
from std_srvs.srv import Trigger, TriggerResponse, TriggerRequest
from message_filters import ApproximateTimeSynchronizer
from dataset_recorder.camera import DualKinectRecorder
from dataset_recorder.utils import map_subinfo_to_idx


class CameraRecorder:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

        self.sync_rate: float = rospy.get_param("~sync_rate", 10.0)
        self.sync_period = rospy.Duration(1.0 / self.sync_rate)
        self.queue_size = rospy.get_param("~queue_size", 10)
        self.slop = rospy.get_param("~slop", 0.1)
        self.last_sync_time = rospy.Time(0)

        self.frame_idx = 0
        self.clip_name = None
        self.clip_dir = None
        self.is_recording = False

        self.recorder = DualKinectRecorder(output_dir)
        self.sub, self.sub_info = [], []
        self.recorder.setup(self.sub, self.sub_info)
        self.msg_idx_map = map_subinfo_to_idx(self.sub_info)
        self.recorder.set_msg_idx_map(self.msg_idx_map)

        self.ts = ApproximateTimeSynchronizer(self.sub, queue_size=10, slop=0.1)
        self.ts.registerCallback(self.sync_callback)

        self.record_service = rospy.Service("~start", Trigger, self.start_record)
        self.stop_service = rospy.Service("~stop", Trigger, self.stop_record)
        rospy.loginfo("Camera recorder is ready!")

    def start_record(self, req: TriggerRequest) -> TriggerResponse:
        if self.is_recording:
            return TriggerResponse(success=False, message="Already recording")

        self.is_recording = True
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.frame_idx = 0

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
        self.frame_idx = 0
        self.clip_name = None
        self.clip_dir = None

        return TriggerResponse(
            success=True,
            message=f"Stopped recording. Recorded {self.frame_idx} frames.",
        )

    def sync_callback(self, *msgs):
        if not self.is_recording:
            return

        current_time = rospy.Time.now()
        interval = current_time - self.last_sync_time
        # if interval < self.sync_period:
        #     return

        self.last_sync_time = current_time

        start_ts = time.perf_counter()
        output_file = self.msgs_dir / f"{self.frame_idx:06d}.pkl"
        data = {
            "timestamp": current_time.to_sec(),
            "msgs": msgs,
            "msg_idx_map": self.msg_idx_map,
        }
        with output_file.open("wb") as f:
            pickle.dump(data, f)
        rospy.loginfo(f"{self.frame_idx}  {interval.to_sec()*1000} ms")
        self.frame_idx += 1


if __name__ == "__main__":
    try:
        rospy.init_node("camera_recorder")
        output_dir = Path(rospy.get_param("~save_dir", "/tmp/dataset_recordings"))

        recorder = CameraRecorder(output_dir)

        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as main_e:
        rospy.logfatal(f"Unhandled exception in main: {main_e}", exc_info=True)
