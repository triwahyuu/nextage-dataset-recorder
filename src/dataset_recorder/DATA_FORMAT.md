# Data Format Reference

This document describes the data formats at each stage of the dataset pipeline:

```
Raw Recording          Processed Dataset         Training-Ready Zarr
(online, ROS node)     (offline, GPU)            (offline, GPU)

msgs/*.pkl         --> rgb/*.png              --> data/left_img
attributes.pkl         depth/*.png                data/right_img
                       depth/*_registered.png     data/left_depth
                       point_cloud/*.npy          data/right_depth
                                                  data/left_pcd
                                                  data/right_pcd
                                                  data/action
                                                  data/state
                                                  meta/episode_ends
```

---

## Stage 1: Raw Recorded Dataset

Output of the `dataset_recorder` ROS node. Contains raw pickled ROS messages and a metadata file.

### Directory Structure

```
<save_dir>/
└── <clip_id>/                  # microsecond timestamp (e.g., "1709123456789012")
    ├── attributes.pkl          # metadata: camera params, TF poses, frame info, task info
    └── msgs/
        ├── 000000.pkl          # pickled tuple of 6 synchronized ROS messages
        ├── 000001.pkl
        ├── 000002.pkl
        └── ...
```

### `msgs/<frame_id>.pkl`

Each file is a Python `pickle` containing a **tuple** of 6 synchronized ROS messages. The messages are ordered according to the `message_idx_map` stored in `attributes.pkl`.

**Default message order:**

| Index | Key in `message_idx_map` | ROS Message Type | Description |
|---|---|---|---|
| 0 | `kinect_left/image` | `sensor_msgs/Image` | Left camera RGB image |
| 1 | `kinect_left/depth` | `sensor_msgs/Image` | Left camera depth image |
| 2 | `kinect_right/image` | `sensor_msgs/Image` | Right camera RGB image |
| 3 | `kinect_right/depth` | `sensor_msgs/Image` | Right camera depth image |
| 4 | `robot_state/joint` | `sensor_msgs/JointState` | Robot joint states |
| 5 | `robot_state/gripper` | `robot_teleop/RobotBoolState` | Gripper boolean states |

**Loading example:**

```python
import pickle
from cv_bridge import CvBridge

bridge = CvBridge()

with open("msgs/000000.pkl", "rb") as f:
    msgs = pickle.load(f)  # tuple of 6 ROS messages

# Extract RGB image (index 0 = kinect_left/image)
rgb_msg = msgs[0]
rgb_image = bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")

# Extract depth image (index 1 = kinect_left/depth)
depth_msg = msgs[1]
depth_image = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")

# Extract joint state (index 4)
joint_msg = msgs[4]
print(joint_msg.name)      # list of joint names
print(joint_msg.position)  # list of joint positions (radians)

# Extract gripper state (index 5)
gripper_msg = msgs[5]
print(gripper_msg.left_gripper)   # bool
print(gripper_msg.right_gripper)  # bool
```

### `attributes.pkl` Schema

The attributes file is a pickled Python dictionary with the following structure. It can also be stored as `attributes.json` (same schema, JSON-serialized).

```python
{
    # ─── Clip identification ───
    "clip_id": str,
    # Microsecond timestamp string, e.g. "1709123456789012"
    # Same as the clip directory name

    # ─── Recording metadata ───
    "metadata": {
        "total_frames": int,
        # Number of frames recorded in this clip

        "duration_sec": float,
        # Total recording duration in seconds

        "start_time": str,
        # Recording start time, format "YYYYMMDD_HHMMSS"

        "end_time": str,
        # Recording end time, format "YYYYMMDD_HHMMSS"
    },

    # ─── Sensor and robot attributes ───
    "attributes": {

        # ─── Camera attributes (one entry per camera) ───
        "camera": {
            "left": {  # (identical structure for "right")

                # ─── Camera pose transforms ───
                # Each transform is looked up from the TF tree at recording start.
                "cam_pose": {

                    "camera2world": {
                        # Transform from camera base to world (WAIST) frame
                        "base_frame_id": str,   # e.g. "WAIST"
                        "frame_id": str,        # e.g. "kinect_left_camera_base"
                        "translation": {
                            "x": float,  # meters
                            "y": float,
                            "z": float,
                        },
                        "rotation": {
                            "x": float,  # quaternion (x, y, z, w)
                            "y": float,
                            "z": float,
                            "w": float,
                        },
                    },

                    "depth2camera": {
                        # Transform from depth sensor to camera base
                        # Same nested structure as camera2world
                    },

                    "rgb2camera": {
                        # Transform from RGB sensor to camera base
                    },

                    "depth2rgb": {
                        # Transform from depth sensor to RGB sensor
                        # Used in post-processing to register depth to RGB
                    },

                    "rgb2world": {
                        # Transform from RGB sensor directly to world frame
                        # Used in post-processing to transform point clouds
                    },
                },

                # ─── Camera intrinsics ───
                "cam_info": {
                    "rgb": {
                        "height": int,
                        # Image height in pixels (e.g. 1080)

                        "width": int,
                        # Image width in pixels (e.g. 1920)

                        "distortion_model": str,
                        # e.g. "rational_polynomial"

                        "D": [float, ...],
                        # Distortion coefficients (length depends on model,
                        # typically 8 for rational_polynomial)

                        "K": [float, float, float,
                              float, float, float,
                              float, float, float],
                        # 3x3 intrinsic camera matrix, row-major (9 elements)
                        # [fx,  0, cx,
                        #   0, fy, cy,
                        #   0,  0,  1]

                        "R": [float, ...],
                        # 3x3 rectification matrix, row-major (9 elements)

                        "P": [float, ...],
                        # 3x4 projection matrix, row-major (12 elements)

                        "binning_x": int,
                        # Binning factor (typically 0)

                        "binning_y": int,

                        "roi": {
                            "x_offset": int,
                            "y_offset": int,
                            "height": int,
                            "width": int,
                            "do_rectify": bool,
                        },
                    },

                    "depth": {
                        # Same structure as "rgb" but for the depth sensor
                        # Typically different resolution (e.g. 640x576 for NFOV)
                    },
                },

                "pcd_frame": str,
                # TF frame used for point clouds
                # e.g. "kinect_left_rgb_camera_link"
            },

            "right": {
                # Same structure as "left"
            },
        },

        # ─── Robot attributes ───
        "robot": {
            "joint_names": [str, ...],
            # List of joint names from the robot driver
            # e.g. ["CHEST_JOINT0", "HEAD_JOINT0", "HEAD_JOINT1",
            #        "LARM_JOINT0", ..., "RARM_JOINT5"]

            "left_eef_tf": str,
            # TF frame name for left end-effector
            # Default: "LARM_LINK_EEF"

            "right_eef_tf": str,
            # TF frame name for right end-effector
            # Default: "RARM_LINK_EEF"

            "world_tf": str,
            # World/base TF frame name
            # Default: "WAIST"
        },

        # ─── Message index map ───
        "message_idx_map": {
            "kinect_left/image": 0,
            "kinect_left/depth": 1,
            "kinect_right/image": 2,
            "kinect_right/depth": 3,
            "robot_state/joint": 4,
            "robot_state/gripper": 5,
        },
        # Maps subscriber info names to indices in the pickled message tuples.
        # Use this to find the correct message in msgs/*.pkl files.
    },

    # ─── Task info ───
    "task_info": {
        "target_object": str,
        # The object the robot should manipulate, e.g. "tomato"

        "available_objects": [str, ...],
        # All objects present in the scene
        # e.g. ["tomato", "lemon", "tennis ball", ...]
    },

    # ─── Per-frame info ───
    "frame_info": [
        {
            "frame_id": str,
            # Zero-padded 6-digit frame index
            # e.g. "000000", "000001", ...

            "timestamp": float,
            # Frame capture time as rospy.Time.now().to_sec()
            # Unix timestamp in seconds with fractional part

            "robot_states": {
                "joint": {
                    "pos": [float, ...],
                    # Joint positions in radians
                    # Length matches joint_names list

                    "vel": [float, ...],
                    # Joint velocities in rad/s

                    "effort": [float, ...],
                    # Joint efforts/torques
                },

                "left_eef": [float, ...],
                # Left end-effector pose as a FLATTENED 4x4 homogeneous
                # transformation matrix (16 elements, row-major).
                # Transform: WAIST -> LARM_LINK_EEF
                #
                # To reconstruct:
                #   tf_mat = np.array(left_eef).reshape(4, 4)
                #   position = tf_mat[:3, 3]          # (x, y, z) in meters
                #   rotation = tf_mat[:3, :3]          # 3x3 rotation matrix
                #
                # Matrix layout:
                #   [r00, r01, r02, tx,
                #    r10, r11, r12, ty,
                #    r20, r21, r22, tz,
                #      0,   0,   0,  1]

                "right_eef": [float, ...],
                # Right end-effector pose, same format as left_eef
                # Transform: WAIST -> RARM_LINK_EEF

                "left_gripper": bool,
                # True = gripper closed/active, False = open/inactive

                "right_gripper": bool,
                # Same for right gripper
            },

            "messages_path": str,
            # Relative path to the pickled messages file
            # e.g. "msgs/000000.pkl"
        },
        # ... one entry per recorded frame
    ],
}
```

**Loading example:**

```python
import pickle
import numpy as np

with open("attributes.pkl", "rb") as f:
    attrs = pickle.load(f)

# Access camera intrinsics
left_K = np.array(attrs["attributes"]["camera"]["left"]["cam_info"]["rgb"]["K"]).reshape(3, 3)
print(f"Left RGB intrinsic matrix:\n{left_K}")

# Access first frame's robot state
frame = attrs["frame_info"][0]
left_eef_mat = np.array(frame["robot_states"]["left_eef"]).reshape(4, 4)
print(f"Left EEF position: {left_eef_mat[:3, 3]}")
print(f"Left gripper closed: {frame['robot_states']['left_gripper']}")

# Get the message index map
idx_map = attrs["attributes"]["message_idx_map"]
print(f"Message indices: {idx_map}")
```

---

## Stage 2: Processed Dataset

Output of `scripts/dataset_processor.py`. Extracts images from raw pickled messages, generates registered depth maps and colored point clouds.

### Directory Structure

```
<clip_dir>/                             # or <output_dir>/<clip_name>/ if -o specified
├── attributes.pkl                      # (unchanged from Stage 1)
├── msgs/                               # (unchanged from Stage 1)
├── rgb/
│   ├── 000000_left.png                 # Left camera RGB image
│   ├── 000000_right.png                # Right camera RGB image
│   ├── 000001_left.png
│   └── ...
├── depth/
│   ├── 000000_left.png                 # Raw depth image
│   ├── 000000_right.png
│   ├── 000000_left_registered.png      # Depth registered to RGB frame
│   ├── 000000_right_registered.png
│   └── ...
└── point_cloud/
    ├── 000000_left_processed.npy       # Colored point cloud in world frame
    ├── 000000_right_processed.npy
    └── ...                             # (or .npz if --compressed-pointcloud)
```

### RGB Images: `rgb/<frame_id>_{left,right}.png`

| Property | Value |
|---|---|
| Format | PNG |
| Color space | BGR (from cv_bridge with `"bgr8"` encoding) |
| Resolution | Original camera resolution (e.g. 1920x1080 for 1080P) |
| Dtype | uint8, 3 channels |

These are the raw undistorted RGB images extracted from the pickled `sensor_msgs/Image` messages.

### Raw Depth Images: `depth/<frame_id>_{left,right}.png`

| Property | Value |
|---|---|
| Format | PNG |
| Dtype | uint16 |
| Units | Millimeters |
| Resolution | Original depth camera resolution (e.g. 640x576 for NFOV_UNBINNED) |

Extracted from `sensor_msgs/Image` with `"passthrough"` encoding. A pixel value of 0 typically indicates no valid depth measurement.

### Registered Depth Images: `depth/<frame_id>_{left,right}_registered.png`

| Property | Value |
|---|---|
| Format | PNG |
| Dtype | uint16 |
| Units | Millimeters (depth_in_meters x 1000) |
| Resolution | Same as RGB image resolution |

The depth image is reprojected into the RGB camera's coordinate frame. This means each pixel in the registered depth image corresponds to the same pixel in the RGB image. This is computed during point cloud generation:

1. Depth pixels are unprojected to 3D using depth camera intrinsics
2. 3D points are transformed from depth frame to RGB frame using `depth2rgb` extrinsic
3. Points are projected onto the RGB image plane using RGB camera intrinsics
4. The resulting depth value at each projected pixel is stored

A pixel value of 0 means no valid depth was projected to that location.

### Point Clouds: `point_cloud/<frame_id>_{left,right}_processed.npy`

| Property | Value |
|---|---|
| Format | NumPy `.npy` (or `.npz` if `--compressed-pointcloud`) |
| Shape | `(N, 6)` where N varies per frame |
| Dtype | float64 (or float32 on GPU path) |
| Columns | `[x, y, z, r, g, b]` |

**Column details:**

| Column | Index | Units | Description |
|---|---|---|---|
| x | 0 | meters | X position in WAIST (world) frame |
| y | 1 | meters | Y position in WAIST (world) frame |
| z | 2 | meters | Z position in WAIST (world) frame |
| r | 3 | 0-255 | Red channel (converted from BGR) |
| g | 4 | 0-255 | Green channel |
| b | 5 | 0-255 | Blue channel |

**Notes:**
- N (number of points) varies per frame depending on valid depth pixels
- Points with invalid depth (zero, negative, or non-finite) are excluded
- Points that project outside the RGB image bounds are excluded
- A warning is logged if N < 4% of total image pixels (rgb_height x rgb_width)
- For `.npz` files, access with: `np.load(path)["pcd"]`
- The GPU (PyTorch) path is automatically used when CUDA is available; otherwise falls back to NumPy/CPU

### Processing Pipeline Detail

For each frame, for each camera (left/right):

1. **Extract images** — Decode RGB (`bgr8`) and depth (`passthrough`) from pickled ROS messages using `cv_bridge`
2. **Save raw images** — Write RGB and depth as PNG
3. **Rectify/undistort** — Apply pre-computed undistortion maps (from `cv2.getOptimalNewCameraMatrix` + `cv2.initUndistortRectifyMap`) to both RGB and depth images
4. **Convert depth to meters** — `depth_float = depth_uint16 / 1000.0`
5. **Create pixel grid** — Generate meshgrid of (u, v) pixel coordinates over the depth image
6. **Filter invalid depth** — Remove pixels where depth <= 1e-6 or non-finite
7. **Unproject to 3D** — Using depth camera intrinsics (K_depth): `X = (u - cx) * Z / fx`, `Y = (v - cy) * Z / fy`
8. **Transform depth→RGB** — Apply `depth2rgb` 4x4 extrinsic matrix: `P_rgb = R @ P_depth + t`
9. **Filter behind camera** — Remove points with Z <= 1e-6 in the RGB camera frame
10. **Project onto RGB plane** — Using RGB camera intrinsics (K_rgb): `u_rgb = fx * X / Z + cx`, `v_rgb = fy * Y / Z + cy`
11. **Filter out-of-bounds** — Remove points that project outside `[0, width) x [0, height)`
12. **Sample colors** — Look up BGR pixel values from the rectified RGB image, convert to RGB
13. **Transform to world** — Apply `rgb2world` 4x4 extrinsic matrix to get WAIST-frame coordinates
14. **Save** — Write registered depth as uint16 PNG, write point cloud as `.npy`/`.npz`

**Loading example:**

```python
import numpy as np
import cv2

# Load point cloud
pcd = np.load("point_cloud/000000_left_processed.npy")  # shape (N, 6)
xyz = pcd[:, :3]   # (N, 3) positions in meters, WAIST frame
rgb = pcd[:, 3:6]  # (N, 3) colors in 0-255

# Load images
rgb_img = cv2.imread("rgb/000000_left.png")                  # BGR uint8
depth_raw = cv2.imread("depth/000000_left.png", cv2.IMREAD_UNCHANGED)  # uint16 mm
depth_reg = cv2.imread("depth/000000_left_registered.png", cv2.IMREAD_UNCHANGED)  # uint16 mm

# Convert depth to meters
depth_meters = depth_reg.astype(np.float32) / 1000.0
```

---

## Stage 3: Training-Ready Zarr Dataset

Output of `scripts/convert_real_robot_data_nxa.py`. Converts processed demonstrations into a compressed Zarr archive suitable for training robot learning policies.

### Output Naming Convention

```
<input_dir_name>_<N>demo_<which_cam>_<num_points>.zarr
```

Example: `recordings_10demo_both_4096.zarr`

### Zarr Structure (dual camera mode: `--which_cam both`)

```
<name>.zarr/
├── data/
│   ├── left_img       (total_frames, img_size, img_size, 3)   uint8
│   ├── right_img      (total_frames, img_size, img_size, 3)   uint8
│   ├── left_depth     (total_frames, img_size, img_size)       float64
│   ├── right_depth    (total_frames, img_size, img_size)       float64
│   ├── left_pcd       (total_frames, num_points, 6)            float64
│   ├── right_pcd      (total_frames, num_points, 6)            float64
│   ├── action         (total_frames, 16)                       float32
│   └── state          (total_frames, 16)                       float32
└── meta/
    └── episode_ends   (num_episodes,)                          int64
```

### Zarr Structure (single camera mode: `--which_cam left` or `right`)

```
<name>.zarr/
├── data/
│   ├── img            (total_frames, img_size, img_size, 3)   uint8
│   ├── depth          (total_frames, img_size, img_size)       float64
│   ├── point_cloud    (total_frames, num_points, 6)            float64
│   ├── action         (total_frames, 16)                       float32
│   └── state          (total_frames, 16)                       float32
└── meta/
    └── episode_ends   (num_episodes,)                          int64
```

### Dataset Details

#### `data/left_img` and `data/right_img` (or `data/img`)

| Property | Value |
|---|---|
| Shape | `(total_frames, img_size, img_size, 3)` |
| Dtype | `uint8` |
| Color space | RGB (loaded via `PIL.Image.open(...).convert("RGB")`) |

The full-resolution RGB image is first **cropped** to a workspace region (in pixels), then **resized** to `img_size x img_size` (default 224).

**Crop regions (pixels):**

| Camera | Top-left (x, y) | Bottom-right (x, y) | Crop size (w x h) |
|---|---|---|---|
| Left | (720, 600) | (1100, 880) | 380 x 280 |
| Right | (1047, 320) | (1380, 620) | 333 x 300 |

The crop is applied as `image[y1:y2, x1:x2]` (NumPy row-major indexing), then resized using `torchvision.transforms.functional.resize` on GPU.

#### `data/left_depth` and `data/right_depth` (or `data/depth`)

| Property | Value |
|---|---|
| Shape | `(total_frames, img_size, img_size)` |
| Dtype | `float64` |
| Source | Registered depth images (`depth/<frame_id>_{cam}_registered.png`) |
| Units | Raw uint16 pixel values (millimeters), **not** converted to meters |

The registered depth PNG is loaded, an extra channel dimension is added, resized to `img_size x img_size` using `torchvision.transforms.functional.resize` on GPU, then the channel dimension is squeezed out.

#### `data/left_pcd` and `data/right_pcd` (or `data/point_cloud`)

| Property | Value |
|---|---|
| Shape | `(total_frames, num_points, 6)` |
| Dtype | `float64` |
| Columns | `[x, y, z, r, g, b]` |
| XYZ units | Meters, in WAIST (world) frame |
| RGB range | 0-255 |

Processing steps:

1. **Load** the processed point cloud `.npy`/`.npz` file (shape `(N, 6)`)
2. **3D workspace crop** — retain only points within the camera-specific 3D bounds (see Workspace Bounds below)
3. **Farthest Point Sampling (FPS)** — subsample to exactly `num_points` (default 4096) points using `pytorch3d.ops.sample_farthest_points`. FPS is applied to the XYZ coordinates only; the corresponding RGB values are looked up by the sampled indices.
4. **Recombine** — stack sampled XYZ and their RGB colors into `(num_points, 6)` array

If the cropped point cloud has fewer points than `num_points`, FPS will oversample (repeat points).

#### `data/action`

| Property | Value |
|---|---|
| Shape | `(total_frames, 16)` |
| Dtype | `float32` |

The action vector encodes the **change in end-effector pose** from the current frame to the next frame, plus the current gripper state.

**Layout (16 elements):**

| Index | Field | Description |
|---|---|---|
| 0-2 | `left_trans_delta` | Left EEF translation delta: `next_pos - curr_pos` (meters) |
| 3-6 | `left_rot_delta` | Left EEF rotation delta as quaternion (x, y, z, w). Computed as `R_next @ R_curr^T`, then converted to quaternion via `scipy.spatial.transform.Rotation` |
| 7 | `left_gripper` | Left gripper state (bool cast to float: 0.0 or 1.0) |
| 8-10 | `right_trans_delta` | Right EEF translation delta (same as left) |
| 11-14 | `right_rot_delta` | Right EEF rotation delta quaternion (same as left) |
| 15 | `right_gripper` | Right gripper state (0.0 or 1.0) |

**Special case — last frame of each episode:**
- Translation delta = `[0.0, 0.0, 0.0]`
- Rotation delta = `[0.0, 0.0, 0.0, 1.0]` (identity quaternion)
- Gripper value = current gripper state

**How the rotation delta is computed:**
```python
curr_rot = curr_tf[:3, :3]    # 3x3 rotation matrix from current frame's 4x4 EEF transform
next_rot = next_tf[:3, :3]    # same for next frame
rot_delta = next_rot @ curr_rot.T  # relative rotation
quat = Rotation.from_matrix(rot_delta).as_quat()  # (x, y, z, w)
```

#### `data/state`

| Property | Value |
|---|---|
| Shape | `(total_frames, 16)` |
| Dtype | `float32` |

The state vector encodes the **absolute end-effector pose** and gripper state at each frame.

**Layout (16 elements):**

| Index | Field | Description |
|---|---|---|
| 0-2 | `left_pos` | Left EEF position: translation from 4x4 transform (meters, WAIST frame) |
| 3-6 | `left_rot` | Left EEF orientation as quaternion (x, y, z, w) from the rotation matrix |
| 7 | `left_gripper` | Left gripper state (0.0 or 1.0) |
| 8-10 | `right_pos` | Right EEF position (same as left) |
| 11-14 | `right_rot` | Right EEF orientation quaternion (same as left) |
| 15 | `right_gripper` | Right gripper state (0.0 or 1.0) |

**How pose is extracted:**
```python
tf_mat = np.array(frame["robot_states"]["left_eef"]).reshape(4, 4)
position = tf_mat[:3, 3]                           # (3,) translation in meters
quaternion = Rotation.from_matrix(tf_mat[:3, :3]).as_quat()  # (4,) as (x, y, z, w)
gripper = float(frame["robot_states"]["left_gripper"])       # 0.0 or 1.0
state[:7] = np.concatenate([position, quaternion])
state[7] = gripper
```

#### `meta/episode_ends`

| Property | Value |
|---|---|
| Shape | `(num_episodes,)` |
| Dtype | `int64` |

An array of **cumulative frame counts** marking the end of each episode. Used to determine episode boundaries in the flat frame arrays.

**Example:**
```
episode_ends = [30, 55, 80]
```
- Episode 0: frames 0 to 29 (30 frames)
- Episode 1: frames 30 to 54 (25 frames)
- Episode 2: frames 55 to 79 (25 frames)
- Total frames: 80

### Compression

All Zarr datasets use:
- Compressor: `zarr.Blosc(cname="zstd", clevel=3, shuffle=1)`
- Chunk size: 100 frames per chunk along the frame dimension

### Filtering

Only episodes whose `task_info.target_object` matches the `--target_object` argument are included. Non-matching episodes are skipped with a warning.

### Loading Example

```python
import zarr
import numpy as np

# Open the Zarr dataset
root = zarr.open("recordings_10demo_both_4096.zarr", mode="r")

# Access datasets
left_imgs = root["data"]["left_img"]     # shape (N, 224, 224, 3)
actions = root["data"]["action"]          # shape (N, 16)
states = root["data"]["state"]            # shape (N, 16)
left_pcds = root["data"]["left_pcd"]      # shape (N, 4096, 6)
episode_ends = root["meta"]["episode_ends"][:]  # shape (num_episodes,)

# Get episode boundaries
print(f"Number of episodes: {len(episode_ends)}")
print(f"Total frames: {episode_ends[-1]}")

# Load a single frame
frame_idx = 0
img = left_imgs[frame_idx]               # (224, 224, 3) uint8
action = actions[frame_idx]              # (16,) float32
state = states[frame_idx]               # (16,) float32
pcd = left_pcds[frame_idx]              # (4096, 6) float64

# Parse action vector
left_trans_delta = action[0:3]
left_rot_delta_quat = action[3:7]
left_gripper = action[7]
right_trans_delta = action[8:11]
right_rot_delta_quat = action[11:15]
right_gripper = action[15]

# Parse state vector
left_pos = state[0:3]
left_rot_quat = state[3:7]
left_grip = state[7]

# Iterate over episodes
start = 0
for ep_idx, end in enumerate(episode_ends):
    ep_actions = actions[start:end]
    ep_states = states[start:end]
    print(f"Episode {ep_idx}: frames {start}-{end-1} ({end - start} frames)")
    start = end
```

---

## Workspace Bounds Reference

These bounds are **hardcoded** in `convert_real_robot_data_nxa.py`. They define the region of interest for cropping images (2D) and point clouds (3D) during training data conversion. Update them if camera positions change.

### 3D Workspace Bounds (meters, WAIST frame)

Used for point cloud cropping before farthest point sampling.

| Camera | X min | X max | Y min | Y max | Z min | Z max |
|---|---|---|---|---|---|---|
| Left | 0.23 | 0.50 | -0.12 | 0.12 | -0.165 | -0.03 |
| Right | 0.27 | 0.51 | -0.06 | 0.18 | -0.155 | -0.03 |

### 2D Image Crop Regions (pixels)

Used for RGB image cropping before resizing.

| Camera | Top-left (x, y) | Bottom-right (x, y) | Label |
|---|---|---|---|
| Left | (720, 600) | (1100, 880) | "center" camera position |
| Right | (1047, 320) | (1380, 620) | |

---

## Coordinate Frames Reference

| Frame | Description |
|---|---|
| `WAIST` | Robot base frame, used as the world frame throughout the pipeline. All point clouds and EEF poses are expressed in this frame. |
| `kinect_{left,right}_camera_base` | Camera mount frame. Static transform from WAIST is defined in `camera_base_pose.launch`. |
| `kinect_{left,right}_depth_camera_link` | Depth sensor optical frame. Transform from camera_base is published by the Azure Kinect driver. |
| `kinect_{left,right}_rgb_camera_link` | RGB sensor optical frame. Transform from camera_base is published by the Azure Kinect driver. |
| `LARM_LINK_EEF` | Left arm end-effector frame. Transform from WAIST is published by the robot driver. |
| `RARM_LINK_EEF` | Right arm end-effector frame. Transform from WAIST is published by the robot driver. |

**Key transform chains used in post-processing:**
- `depth2rgb`: depth_camera_link → rgb_camera_link (used to register depth to RGB)
- `rgb2world`: rgb_camera_link → WAIST (used to transform point clouds to world frame)
