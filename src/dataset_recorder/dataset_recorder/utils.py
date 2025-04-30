import rospy

def map_msg_sync(msgs, subs_info):
    received_data = {}
    if len(msgs) != len(subs_info):
        rospy.logwarn_throttle(5.0, f"Mismatch between received messages ({len(msgs)}) and expected subscribers ({len(subs_info)}). Skipping frame.")
        return
    for info, msg in zip(subs_info, msgs):
        received_data[info["name"]] = msg
    return received_data

def map_subinfo_to_idx(subs_info: dict):
    sub_idx = {}
    for idx, info in enumerate(subs_info):
        sub_idx[info["name"]] = idx
    return sub_idx
