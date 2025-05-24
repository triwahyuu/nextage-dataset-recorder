#!/usr/bin/env python

import rospy
import time

from pathlib import Path
from dataset_recorder.camera import TopicSubscriber, KinectRecorder

if __name__ == "__main__":
    try:
        output_dir = Path(rospy.get_param("~save_dir", "/tmp/dataset_recordings"))
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        session_dir = output_dir / timestamp
        session_dir.mkdir(parents=True, exist_ok=True)

        recorder = KinectRecorder(output_dir)
        recorder.reset_output_dir(session_dir)
        sub, sub_info = [], []
        recorder.setup(sub, sub_info)

        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as main_e:
        rospy.logfatal(f"Unhandled exception in main: {main_e}", exc_info=True)
