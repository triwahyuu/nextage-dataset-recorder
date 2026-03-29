# Dataset Recorder

A ROS Noetic workspace for recording synchronized demonstration datasets from dual Azure Kinect cameras and a bimanual Nextage robot. Designed for collecting teleoperated manipulation demonstrations and converting them into training-ready datasets for robot learning.


## Documentation

- **[Package Documentation](src/dataset_recorder/README.md)** — Full usage guide: installation, launching, recording, post-processing, ROS interface (topics, TFs, services, parameters), and configuration.
- **[Data Format Reference](src/dataset_recorder/DATA_FORMAT.md)** — Detailed specification of data formats at each pipeline stage: raw recorded format, processed dataset format, and training-ready Zarr format.


## Pipeline Overview

The system operates in three stages:

1. **Record** (online) — A ROS node synchronizes RGB-D streams from two Kinect cameras with robot joint states and gripper states, saving raw pickled ROS messages per frame along with sensor calibration metadata.

2. **Process** (offline, GPU-accelerated) — Extracts images from pickled messages, undistorts them, registers depth to RGB, and generates colored 3D point clouds in the world coordinate frame.

3. **Convert** (offline) — Transforms processed data into a compressed Zarr archive with workspace-cropped images, farthest-point-sampled point clouds, and end-effector action/state vectors ready for policy training.


## Hardware

- **Cameras:** 2x Azure Kinect DK RGBD Cameras
- **Robot:** Nextage (bimanual humanoid, base frame: `WAIST`)
- **GPU:** NVIDIA GPU with CUDA 11.8 (for post-processing)

## Development Environment

### Option 1: VS Code Dev Container

The repository includes a Dev Container configuration (`.devcontainer/`) with all dependencies pre-installed.

1. Install the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) in VS Code.
2. Open the repository in VS Code.
3. When prompted, click **"Reopen in Container"** — or run the command **Dev Containers: Reopen in Container** from the command palette.

The container will be pulled and configured automatically.

### Option 2: Docker (manual)

Use the pre-built Docker image directly with `triwahyuu/nextagea-dataset-recorder:dev` image:

```bash
docker run -it --gpus=all --privileged --net=host --ipc=host --pid=host \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY \
    -v $(pwd):/workspaces/dataset_recorder \
    triwahyuu/nextagea-dataset-recorder:dev
```

> **Note:** Run `xhost +` on the host before starting the container to enable GUI forwarding.


## Quick Start

```bash
# Build
source /opt/ros/noetic/setup.bash
catkin build && source devel/setup.bash

# Launch cameras and recorder
roslaunch dataset_recorder dual_kinect_camera.launch
roslaunch dataset_recorder dataset_recorder.launch \
    task_config_path:=$(rospack find dataset_recorder)/task_configs/set0_target0.json

# Record a demonstration
rosservice call /dataset_recorder/start "{}"
# ... perform demonstration ...
rosservice call /dataset_recorder/stop "{}"

# Post-process into images and point clouds
python src/dataset_recorder/scripts/dataset_processor.py /path/to/recordings

# Convert to training-ready Zarr
python src/dataset_recorder/scripts/convert_real_robot_data_nxa.py \
    /path/to/recordings /path/to/output
```

## Notes

- Generated point clouds from [dataset_processor.py](src/dataset_recorder/scripts/dataset_processor.py) are in world (WAIST) coordinate frame.
- The devcontainer (`.devcontainer/`) provides a complete development environment with all dependencies pre-installed.
