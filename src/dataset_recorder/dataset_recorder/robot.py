import rospy
import message_filters
import tf2_ros
import tf.transformations as tfs
import numpy as np
from sensor_msgs.msg import JointState

from robot_teleop.msg import RobotBoolState


class RobotStateRecorder:
    """
    Handles recording of robot states: joint state, end-effector pose, gripper states.
    """
    def __init__(self):
        """
        Initializes the recorder.

        Args:
            output_dir (str): The base directory where data will be saved.
        """
        rospy.loginfo(f"Initializing Robot State Recorder...")

        joint_state_topic = rospy.get_param("~robot_state_topic", "/joint_states").lstrip('/')
        self.joint_state_topic: str = f"/{joint_state_topic}"
        grip_state_topic = rospy.get_param("~gripper_state_topic", "/robot_gripper_state").lstrip('/')
        self.gripper_state_topic: str = f"/{grip_state_topic}"

        self.left_eef_tf = rospy.get_param("~left_eef_tf_name", "LARM_LINK_EEF")
        self.right_eef_tf = rospy.get_param("~right_eef_tf_name", "RARM_LINK_EEF")
        self.world_tf = rospy.get_param("~world_tf_name", "WAIST")
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.joint_names = []

        self.joint_state_sub = None
        self.gripper_state_sub = None
        self.msg_idx_map = {}

    def setup(self, subscribers: list, subscriber_info: list):
        """Sets up the ROS subscribers."""
        self.joint_state_sub = message_filters.Subscriber(self.joint_state_topic, JointState)
        self.joint_info_name = f"robot_state/joint"
        subscribers.append(self.joint_state_sub)
        subscriber_info.append({"name": self.joint_info_name, "type": JointState})

        self.gripper_state_sub = message_filters.Subscriber(self.gripper_state_topic, RobotBoolState)
        self.gripper_info_name = f"robot_state/gripper"
        subscribers.append(self.gripper_state_sub)
        subscriber_info.append({"name": self.gripper_info_name, "type": RobotBoolState})

        msg: JointState = rospy.wait_for_message(self.joint_state_topic, JointState, timeout=5)
        self.joint_names = list(msg.name)

    def set_msg_idx_map(self, idx_map: dict):
        self.msg_idx_map = idx_map

    def get_attributes(self):
        return {
            "joint_names": self.joint_names,
            "left_eef_tf": self.left_eef_tf,
            "right_eef_tf": self.right_eef_tf,
            "world_tf": self.world_tf
        }

    def get_pose_tf(self, tf_name: str, base_frame: str) -> tf2_ros.TransformStamped:
        try:
            tf_stamped: tf2_ros.TransformStamped = self.tf_buffer.lookup_transform(
                base_frame, tf_name, rospy.Time(0), rospy.Duration(0.1)
            )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"[RobotStateRecorder] Could not get transform from '{base_frame}' to '{tf_name}': {e}")
            return None
        return tf_stamped

    def get_pose_tf_mat(self, tf_name: str, base_frame: str):
        tfstamped = self.get_pose_tf(tf_name, base_frame)
        if tfs is None:
            return None
        
        translation = [
            tfstamped.transform.translation.x,
            tfstamped.transform.translation.y,
            tfstamped.transform.translation.z
        ]
    
        rotation_quat = [
            tfstamped.transform.rotation.x,
            tfstamped.transform.rotation.y,
            tfstamped.transform.rotation.z,
            tfstamped.transform.rotation.w
        ]

        trans_mat = tfs.translation_matrix(translation)
        rot_mat = tfs.quaternion_matrix(rotation_quat)
        
        # Combine the translation and rotation matrices
        # M = T * R
        tf_matrix = np.dot(trans_mat, rot_mat)
        return list(tf_matrix.flatten())

    def get_robot_state(self, msgs: list, frame_id: str) -> str:
        """Returns robot state as a dictionary."""

        try:
            joint_msg: JointState = msgs[self.msg_idx_map[self.joint_info_name]]
            gripper_msg: RobotBoolState = msgs[self.msg_idx_map[self.gripper_info_name]]

            left_eef = self.get_pose_tf_mat(self.left_eef_tf, self.world_tf)
            right_eef = self.get_pose_tf_mat(self.right_eef_tf, self.world_tf)

            joint_states = {
                "pos": list(joint_msg.position) if joint_msg.position is not None else [],
                "vel": list(joint_msg.velocity) if joint_msg.velocity is not None else [],
                "effort": list(joint_msg.effort) if joint_msg.effort is not None else [],
            }

            state_dict = {
                "joint": joint_states,
                "left_eef": left_eef,
                "right_eef": right_eef,
                "left_gripper": bool(gripper_msg.left_gripper),
                "right_gripper": bool(gripper_msg.right_gripper)
            }
            return state_dict
        except Exception as e:
            rospy.logerr(f"Failed to retrieve robot state {frame_id}: {e}")
            raise


if __name__ == "__main__":
    from dataset_recorder.utils import map_subinfo_to_idx

    rospy.init_node("robot_recorder", anonymous=True)
    recorder = RobotStateRecorder()

    subs, sub_info = [], []
    recorder.setup(subs, sub_info)
    recorder.set_msg_idx_map(map_subinfo_to_idx(sub_info))

    attr = recorder.get_attributes()

    frame_num = 0

    def sync_callback(*msgs):
        global frame_num
        try:
            frame_id = f"{frame_num:06d}"
            robot_states = recorder.get_robot_state(msgs, frame_id)
            frame_num += 1
        except Exception as e:
            rospy.logerr(f"Error saving data: {e}")

    ts = message_filters.ApproximateTimeSynchronizer(
        subs, queue_size=10, slop=0.1
    )
    ts.registerCallback(sync_callback)
    rospy.spin()
