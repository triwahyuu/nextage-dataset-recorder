# Dataset Recorder

A ROS package for recording various sensor data including images, point clouds, and robot states. The recording can be triggered via ROS services.

## Features

- Record data from multiple sensors simultaneously
- Configurable topics for images, point clouds, and robot states
- Service-based trigger to start and stop recording
- Option for synchronized or independent recording of data streams
- Organized storage of recorded data with timestamps

## Installation

Clone this repository into your catkin workspace and build it:

```bash
cd ~/catkin_ws/src
git clone <your-repo-url>/dataset_recorder.git
cd ..
catkin_make
source devel/setup.bash
```

## Usage

### Launch the recorder node

```bash
roslaunch dataset_recorder dataset_recorder.launch
```

### Customize parameters

Edit the launch file or pass parameters on the command line:

```bash
roslaunch dataset_recorder dataset_recorder.launch save_dir:=/path/to/save/data image_topic:=/my/image/topic
```

### Available parameters


### Start and stop recording

To start recording:
```bash
rosservice call /dataset_recorder/start_recording
```

To stop recording:
```bash
rosservice call /dataset_recorder/stop_recording
```

## Data Structure

The recorded data is organized as follows:

```
session_YYYYMMDD_HHMMSS/
├── images/
│   ├── image_timestamp1.jpg
│   ├── image_timestamp2.jpg
│   └── ...
├── pointclouds/
│   ├── pointcloud_timestamp1.bin
│   ├── pointcloud_timestamp2.bin
│   └── ...
└── robot_states/
    ├── state_timestamp1.txt
    ├── state_timestamp2.txt
    └── ...
```

## Dependencies

- ROS (tested with Noetic)
- OpenCV
- cv_bridge
- sensor_msgs
- std_msgs
- std_srvs
