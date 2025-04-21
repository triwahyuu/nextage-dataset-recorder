#!/usr/bin/env python

import rospy

from dataset_recorder.recorder import DatasetRecorder

if __name__ == "__main__":
    try:
        recorder = DatasetRecorder()
        if hasattr(recorder, 'ts'): # Check if initialization was successful
            rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as main_e:
        rospy.logfatal(f"Unhandled exception in main: {main_e}", exc_info=True)
