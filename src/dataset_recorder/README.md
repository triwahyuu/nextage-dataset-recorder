# Dataset Recorder

A ROS Noetic catkin package for recording synchronized sensor data from dual Azure Kinect cameras and a bimanual Nextage robot. The system captures raw ROS messages during teleoperated demonstrations, then post-processes them offline into images, point clouds, and training-ready Zarr datasets for robot learning.

The data pipeline has three stages: **Record** (online, ROS node) → **Process** (offline, GPU-accelerated) → **Convert** (to Zarr for ML training).

For detailed data format specifications, see [DATA_FORMAT.md](DATA_FORMAT.md).

## Prerequisites

### Hardware

- 2x Azure Kinect DK cameras
  - Left camera serial: `000153513712`
  - Right camera serial: `000231113712`
- Nextage robot (or any bimanual robot providing the required topics/TFs)
- NVIDIA GPU with CUDA 11.8 support (for post-processing)

### Software

- Ubuntu 20.04
- ROS Noetic
- Azure Kinect ROS Driver (included as git submodule in `src/Azure_Kinect_ROS_Driver/`)
- `robot_teleop` package (provides `RobotBoolState` message; included in `src/robot_teleop/`)
- Robot driver providing `/joint_states` and EEF TF transforms
- Python packages: `torch`, `torchvision`, `pytorch3d`, `zarr`, `scipy`, `opencv-python`, `cv_bridge`, `tqdm`, `termcolor`, `Pillow`
- `ffmpeg` (for video conversion)

The included devcontainer sets up all software dependencies automatically.

## Installation & Build

```bash
cd /workspaces/dataset_recorder
source /opt/ros/noetic/setup.bash
catkin build
source devel/setup.bash
```

If using the devcontainer, the `postCreateCommand.sh` script handles building, submodule init, and udev rules automatically.

## Quick Start

```bash
# 1. Launch dual Kinect cameras + static TF transforms
roslaunch dataset_recorder dual_kinect_camera.launch

# 2. Launch the recorder node (task_config_path is required)
roslaunch dataset_recorder dataset_recorder.launch \
    task_config_path:=$(rospack find dataset_recorder)/task_configs/set0_target0.json

# 3. Start/stop recording via ROS services
rosservice call /dataset_recorder/start "{}"
# ... perform demonstration ...
rosservice call /dataset_recorder/stop "{}"

# 4. Post-process raw recordings into images and point clouds
python src/dataset_recorder/scripts/dataset_processor.py /path/to/recordings

# 5. Convert to training-ready Zarr dataset
python src/dataset_recorder/scripts/convert_real_robot_data_nxa.py \
    /path/to/recordings /path/to/output
```

## Architecture

`DatasetRecorder` is the main ROS node. It composes two sub-recorders:

- **`DualKinectRecorder`** — manages two `KinectRecorder` instances (left/right), each subscribing to RGB and depth image topics. At startup, reads camera intrinsics from `CameraInfo` topics and looks up camera-to-world TF transforms.
- **`RobotStateRecorder`** — subscribes to joint state and gripper state topics. On each frame, looks up left/right end-effector TF transforms.

Both sub-recorders register their `message_filters.Subscriber` instances into a shared list, which is passed to a single `ApproximateTimeSynchronizer` (6 topics, configurable `queue_size` and `slop`). On each synchronized callback:

1. The entire message tuple (6 ROS messages) is pickled to `msgs/<frame_id>.pkl`
2. Robot state (joints, EEF poses, gripper states) is extracted and stored in `frame_info`

When recording stops, `attributes.pkl` is written with camera intrinsics/extrinsics, robot attributes, the message index map, frame info, and task info.

Post-processing (`dataset_processor.py`) unpacks the pickled messages and generates rectified images, registered depth maps, and colored point clouds in the world frame. It has both a NumPy/CPU and a PyTorch/GPU code path, auto-selected based on CUDA availability.

## Usage

### Launching the Cameras

```bash
roslaunch dataset_recorder dual_kinect_camera.launch
```

This launch file includes `camera_base_pose.launch` (static camera-to-WAIST transforms) and starts two Azure Kinect driver instances.

| Argument | Default | Description |
|---|---|---|
| `kinect_left_sn` | `000165213712` | Left Kinect serial number |
| `kinect_right_sn` | `000231113712` | Right Kinect serial number |
| `fps` | `15` | Camera FPS (valid: 5, 15, 30) |
| `color_resolution` | `1080P` | Color resolution (720P, 1080P, 1440P, 1536P, 2160P, 3072P) |

**Note:** The camera launch is currently commented out inside `dataset_recorder.launch` and must be launched separately.

### Launching the Recorder

```bash
roslaunch dataset_recorder dataset_recorder.launch task_config_path:=/path/to/config.json
```

| Argument | Default | Description |
|---|---|---|
| `task_config_path` | **(required)** | Path to task config JSON |
| `save_dir` | `/workspaces/dataset-recorder/recordings` | Recording output directory |

### Recording Demonstrations

Start a new recording clip:
```bash
rosservice call /dataset_recorder/start "{}"
```

Stop recording and save `attributes.pkl`:
```bash
rosservice call /dataset_recorder/stop "{}"
```

Each clip is saved to `<save_dir>/<microsecond_timestamp>/`.

### Post-Processing: Generating Point Clouds

```bash
python scripts/dataset_processor.py <dataset_dir> [options]
```

| Option | Default | Description |
|---|---|---|
| `--output-dir`, `-o` | same as input | Output directory for generated data |
| `--num-demo`, `-n` | all | Limit number of episodes to process |
| `--no-pointcloud` | off | Only extract images, skip point cloud generation |
| `--compressed-pointcloud` | off | Save point clouds as `.npz` instead of `.npy` |

### Converting to Training Data (Zarr)

Requires processed data (RGB, registered depth, point clouds) to exist first.

```bash
python scripts/convert_real_robot_data_nxa.py <input_dir> <output_dir> [options]
```

| Option | Default | Description |
|---|---|---|
| `--num-demo`, `-n` | all | Limit number of episodes |
| `--which_cam`, `-c` | `both` | Camera selection: `left`, `right`, or `both` |
| `--num_points`, `-p` | `4096` | Points per cloud after farthest point sampling |
| `--img_size` | `224` | Image resize target (square) |
| `--target_object` | `tomato` | Filter demos by target object |

Output file is named: `<input_dir_name>_<N>demo_<cam>_<points>.zarr`

### Utility Scripts

**Create review videos from RGB frames:**
```bash
python scripts/convert_videos.py <root_dir> [-r framerate] [-c crf] [-p preset]
```
Produces `left.mp4` and `right.mp4` per episode from `rgb/%06d_{left,right}.png` sequences.

**Remove processed directories to reclaim space:**
```bash
python scripts/cleanup_data.py <path>
```
Removes `point_cloud/`, `rgb/`, `depth/` from all episode subdirectories (interactive confirmation).

**Visualize a point cloud in RViz:**
```bash
roslaunch dataset_recorder visualize_pointcloud.launch pointcloud_file:=/path/to/cloud.npy
```

### Camera-Only Recorder

A standalone camera recorder node (no robot state) is also available:

```bash
rosrun dataset_recorder camera_recorder_node.py
```

Services: `~start` and `~stop` (`std_srvs/Trigger`). Saves frames as `msgs/<idx>.pkl` with each pickle containing `{"timestamp": float, "msgs": tuple, "msg_idx_map": dict}`.

## ROS Interface

### Subscribed Topics (Synchronized)

All 6 topics must be publishing for the `ApproximateTimeSynchronizer` to fire.

| Topic | Type | Description |
|---|---|---|
| `/kinect_left/rgb/image_raw` | `sensor_msgs/Image` | Left camera RGB image |
| `/kinect_left/depth/image_raw` | `sensor_msgs/Image` | Left camera depth image |
| `/kinect_right/rgb/image_raw` | `sensor_msgs/Image` | Right camera RGB image |
| `/kinect_right/depth/image_raw` | `sensor_msgs/Image` | Right camera depth image |
| `/joint_states` | `sensor_msgs/JointState` | Robot joint positions, velocities, efforts |
| `/robot_gripper_state` | `robot_teleop/RobotBoolState` | Left/right gripper boolean states |

### One-Shot Reads at Startup

These topics are read once via `rospy.wait_for_message` (1-second timeout) during node initialization:

| Topic | Type | Purpose |
|---|---|---|
| `/kinect_left/rgb/camera_info` | `sensor_msgs/CameraInfo` | Left RGB camera intrinsics |
| `/kinect_left/depth/camera_info` | `sensor_msgs/CameraInfo` | Left depth camera intrinsics |
| `/kinect_right/rgb/camera_info` | `sensor_msgs/CameraInfo` | Right RGB camera intrinsics |
| `/kinect_right/depth/camera_info` | `sensor_msgs/CameraInfo` | Right depth camera intrinsics |
| `/joint_states` | `sensor_msgs/JointState` | Capture joint names list |

### Required TF Transforms

| Parent Frame | Child Frame | Source |
|---|---|---|
| `WAIST` | `kinect_left_camera_base` | `camera_base_pose.launch` (static) |
| `WAIST` | `kinect_right_camera_base` | `camera_base_pose.launch` (static) |
| `kinect_{left,right}_camera_base` | `kinect_{left,right}_depth_camera_link` | Azure Kinect driver (URDF) |
| `kinect_{left,right}_camera_base` | `kinect_{left,right}_rgb_camera_link` | Azure Kinect driver (URDF) |
| `WAIST` | `LARM_LINK_EEF` | Robot driver |
| `WAIST` | `RARM_LINK_EEF` | Robot driver |

### Services

| Service | Type | Description |
|---|---|---|
| `/dataset_recorder/start` | `std_srvs/Trigger` | Start recording a new clip |
| `/dataset_recorder/stop` | `std_srvs/Trigger` | Stop recording, save attributes.pkl |

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `~save_dir` | `/tmp/dataset_recordings` | Base save directory |
| `~sync_rate` | `10.0` | Target sync rate in Hz (rate-limiting currently commented out) |
| `~queue_size` | `10` | ApproximateTimeSynchronizer queue size |
| `~slop` | `0.1` | ApproximateTimeSynchronizer slop (seconds) |
| `~task_config_path` | **(required)** | Path to task config JSON file |
| `~image_topic` | `/rgb/image_raw` | RGB image topic suffix (appended to camera namespace) |
| `~depth_topic` | `/depth/image_raw` | Depth image topic suffix |
| `~rgb_caminfo_topic` | `/rgb/camera_info` | RGB CameraInfo topic suffix |
| `~depth_caminfo_topic` | `/depth/camera_info` | Depth CameraInfo topic suffix |
| `~robot_state_topic` | `/joint_states` | Joint state topic |
| `~gripper_state_topic` | `/robot_gripper_state` | Gripper state topic |
| `~left_eef_tf_name` | `LARM_LINK_EEF` | Left end-effector TF frame name |
| `~right_eef_tf_name` | `RARM_LINK_EEF` | Right end-effector TF frame name |
| `~world_tf_name` | `WAIST` | World/base TF frame name |

## Task Configuration

The task config is a JSON file specifying the target object and available objects in the scene. It is required by the `~task_config_path` parameter.

**Schema:**
```json
{
    "target_object": "string",
    "available_objects": ["string", ...]
}
```

**Example** (`task_configs/set0_target0.json`):
```json
{
    "target_object": "tomato",
    "available_objects": [
        "tomato", "lemon", "tennis ball", "big clamp",
        "glue bottle", "spoon", "fork", "sponge"
    ]
}
```

## Custom Messages

### `robot_teleop/RobotBoolState`

```
std_msgs/Header header
bool servo
bool left_gripper
bool right_gripper
```

Published on `/robot_gripper_state` by the teleop node. The `left_gripper` and `right_gripper` fields indicate gripper open/close state.
