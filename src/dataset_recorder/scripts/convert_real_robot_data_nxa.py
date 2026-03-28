import zarr
import pickle
import tqdm
import json
import numpy as np
import torch
import pytorch3d.ops as torch3d_ops
import torchvision
import argparse

from termcolor import cprint
from pathlib import Path
from PIL import Image
from scipy.spatial.transform import Rotation as R

WORKSPACE_3D_RIGHT = [[0.27, 0.51], [-0.06, 0.18], [-0.155, -0.03]]
WORKSPACE_3D_LEFT = [[0.23, 0.50], [-0.12, 0.12], [-0.165, -0.03]]
WORKSPACE_2D_RIGHT = ((1047, 320), (1380, 620))
# WORKSPACE_2D_LEFT = ((775, 375), (1065, 660))
WORKSPACE_2D_LEFT = ((720, 600), (1100, 880))  # center


def farthest_point_sampling(points, num_points=1024, use_cuda=True):
    K = [num_points]
    if use_cuda:
        points = torch.from_numpy(points).cuda()
        sampled_points, indices = torch3d_ops.sample_farthest_points(
            points=points.unsqueeze(0), K=K
        )
        sampled_points = sampled_points.squeeze(0)
        sampled_points = sampled_points.cpu().numpy()
    else:
        points = torch.from_numpy(points)
        sampled_points, indices = torch3d_ops.sample_farthest_points(
            points=points.unsqueeze(0), K=K
        )
        sampled_points = sampled_points.squeeze(0)
        sampled_points = sampled_points.numpy()

    return sampled_points, indices


def preprocess_point_cloud(
    points: np.ndarray, num_points: int, which_cam: str, use_cuda=True
) -> np.ndarray:
    worspace3d = WORKSPACE_3D_RIGHT if which_cam == "right" else WORKSPACE_3D_LEFT
    # crop
    points = points[
        np.where(
            (points[..., 0] > worspace3d[0][0])
            & (points[..., 0] < worspace3d[0][1])
            & (points[..., 1] > worspace3d[1][0])
            & (points[..., 1] < worspace3d[1][1])
            & (points[..., 2] > worspace3d[2][0])
            & (points[..., 2] < worspace3d[2][1])
        )
    ]

    points_xyz = points[..., :3]
    points_xyz, sample_indices = farthest_point_sampling(
        points_xyz, num_points, use_cuda
    )
    sample_indices = sample_indices.cpu()
    points_rgb = points[sample_indices, 3:][0]
    points = np.hstack((points_xyz, points_rgb))
    return points


def preproces_image(image: np.ndarray, img_size: int) -> np.ndarray:
    image = image.astype(np.float32)
    image = torch.from_numpy(image).cuda()
    image = image.permute(2, 0, 1)  # HxWx4 -> 4xHxW
    image = torchvision.transforms.functional.resize(image, (img_size, img_size))
    image = image.permute(1, 2, 0)  # 4xHxW -> HxWx4
    image = image.cpu().numpy()
    return image


def preproces_rgb_image(image: np.ndarray, which_cam: str, img_size: int) -> np.ndarray:
    ws = WORKSPACE_2D_RIGHT if which_cam == "right" else WORKSPACE_2D_LEFT
    (x1, y1), (x2, y2) = ws

    image_cropped = image[y1:y2, x1:x2]
    img_res = preproces_image(image_cropped, img_size)
    return img_res


def get_pose_delta(curr_tf: np.ndarray, next_tf: np.ndarray = None) -> np.ndarray:
    """
    Calculates the change in pose from curr_tf to next_tf.

    Args:
        curr_tf (np.ndarray): Current 4x4 transformation matrix.
        next_tf (np.ndarray): Next 4x4 transformation matrix.

    Returns:
        np.ndarray: pose as vector of 7 elements [translation, quaternion]
    """
    if next_tf is None:
        # last frame has no next pose, delta is zero
        vec = np.zeros(7, dtype=np.float32)
        vec[6] = 1.0
        return vec

    trans_delta = next_tf[:3, 3] - curr_tf[:3, 3]

    # rotation delta: R_delta = R_curr @ R_prev.T
    curr_rot = curr_tf[:3, :3]
    next_rot = next_tf[:3, :3]
    rot_delta = next_rot @ curr_rot.T

    vec = np.zeros(7, dtype=np.float32)
    vec[:3] = trans_delta
    vec[3:] = R.from_matrix(rot_delta).as_quat()
    return vec


def get_pose_vec(tf: np.ndarray):
    vec = np.zeros(7, dtype=np.float32)
    vec[:3] = tf[:3, 3]
    vec[3:] = R.from_matrix(tf[:3, :3]).as_quat()
    return vec


class DataConverter:
    def __init__(
        self,
        target_object,
        which_cam,
        save_data_path,
        overwrite_output,
        chunk_size,
        num_points,
        img_size,
    ):
        self.target_object = target_object
        self.which_cam = which_cam
        self.save_data_path = save_data_path
        self.chunk_size = chunk_size
        self.num_points = num_points
        self.img_size = img_size

        self.save_data_path.mkdir(parents=True, exist_ok=True)
        self.zarr_root = zarr.group(self.save_data_path, overwrite=overwrite_output)
        self.zarr_data = self.zarr_root.create_group("data")
        self.zarr_meta = self.zarr_root.create_group("meta")

        compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)

        img_shape = (self.img_size, self.img_size, 3)
        pc_shape = (self.num_points, 6)
        depth_shape = (self.img_size, self.img_size)
        action_shape = (16,)
        state_shape = (16,)

        if self.which_cam == "both":
            self.z_img_left = self.zarr_data.create_dataset(
                "left_img",
                shape=(0,) + img_shape,
                chunks=(self.chunk_size,) + img_shape,
                dtype="uint8",
                compressor=compressor,
                overwrite=True,
            )
            self.z_img_right = self.zarr_data.create_dataset(
                "right_img",
                shape=(0,) + img_shape,
                chunks=(self.chunk_size,) + img_shape,
                dtype="uint8",
                compressor=compressor,
                overwrite=True,
            )
            self.z_depth_left = self.zarr_data.create_dataset(
                "left_depth",
                shape=(0,) + depth_shape,
                chunks=(self.chunk_size,) + depth_shape,
                dtype="float64",
                compressor=compressor,
                overwrite=True,
            )
            self.z_depth_right = self.zarr_data.create_dataset(
                "right_depth",
                shape=(0,) + depth_shape,
                chunks=(self.chunk_size,) + depth_shape,
                dtype="float64",
                compressor=compressor,
                overwrite=True,
            )
            self.z_pc_left = self.zarr_data.create_dataset(
                "left_pcd",
                shape=(0,) + pc_shape,
                chunks=(self.chunk_size,) + pc_shape,
                dtype="float64",
                compressor=compressor,
                overwrite=True,
            )
            self.z_pc_right = self.zarr_data.create_dataset(
                "right_pcd",
                shape=(0,) + pc_shape,
                chunks=(self.chunk_size,) + pc_shape,
                dtype="float64",
                compressor=compressor,
                overwrite=True,
            )
        else:
            self.z_img = self.zarr_data.create_dataset(
                "img",
                shape=(0,) + img_shape,
                chunks=(self.chunk_size,) + img_shape,
                dtype="uint8",
                compressor=compressor,
                overwrite=True,
            )
            self.z_pc = self.zarr_data.create_dataset(
                "point_cloud",
                shape=(0,) + pc_shape,
                chunks=(self.chunk_size,) + pc_shape,
                dtype="float64",
                compressor=compressor,
                overwrite=True,
            )
            self.z_depth = self.zarr_data.create_dataset(
                "depth",
                shape=(0,) + depth_shape,
                chunks=(self.chunk_size,) + depth_shape,
                dtype="float64",
                compressor=compressor,
                overwrite=True,
            )

        self.z_action = self.zarr_data.create_dataset(
            "action",
            shape=(0,) + action_shape,
            chunks=(self.chunk_size,) + action_shape,
            dtype="float32",
            compressor=compressor,
            overwrite=True,
        )
        self.z_state = self.zarr_data.create_dataset(
            "state",
            shape=(0,) + state_shape,
            chunks=(self.chunk_size,) + state_shape,
            dtype="float32",
            compressor=compressor,
            overwrite=True,
        )

        if self.which_cam == "both":
            self.img_left_batch_buffer = []
            self.img_right_batch_buffer = []
            self.depth_left_batch_buffer = []
            self.depth_right_batch_buffer = []
            self.pc_left_batch_buffer = []
            self.pc_right_batch_buffer = []
        else:
            self.img_batch_buffer = []
            self.pc_batch_buffer = []
            self.depth_batch_buffer = []
        self.action_batch_buffer = []
        self.state_batch_buffer = []

        self.total_count = 0
        self.episode_ends_arrays = []

    def _flush_buffers_to_zarr(self):
        if not self.action_batch_buffer:  # If one buffer is empty, all should be
            return

        if self.which_cam == "both":
            self.z_img_left.append(
                np.stack(self.img_left_batch_buffer, axis=0).astype(np.uint8)
            )
            self.z_img_right.append(
                np.stack(self.img_right_batch_buffer, axis=0).astype(np.uint8)
            )
            self.z_depth_left.append(np.stack(self.depth_left_batch_buffer, axis=0))
            self.z_depth_right.append(np.stack(self.depth_right_batch_buffer, axis=0))
            self.z_pc_left.append(np.stack(self.pc_left_batch_buffer, axis=0))
            self.z_pc_right.append(np.stack(self.pc_right_batch_buffer, axis=0))
            self.img_left_batch_buffer.clear()
            self.img_right_batch_buffer.clear()
            self.depth_left_batch_buffer.clear()
            self.depth_right_batch_buffer.clear()
            self.pc_left_batch_buffer.clear()
            self.pc_right_batch_buffer.clear()
        else:
            self.z_img.append(np.stack(self.img_batch_buffer, axis=0).astype(np.uint8))
            self.z_pc.append(np.stack(self.pc_batch_buffer, axis=0))
            self.z_depth.append(np.stack(self.depth_batch_buffer, axis=0))
            self.img_batch_buffer.clear()
            self.pc_batch_buffer.clear()
            self.depth_batch_buffer.clear()

        self.z_action.append(np.stack(self.action_batch_buffer, axis=0))
        self.z_state.append(np.stack(self.state_batch_buffer, axis=0))
        self.action_batch_buffer.clear()
        self.state_batch_buffer.clear()

    def process_demo_directory(self, demo_dir: Path):
        attrs = self._load_attributes(demo_dir)

        sample_target_object = attrs["task_info"]["target_object"]
        if sample_target_object != self.target_object:
            cprint(
                f"Skipping target {sample_target_object}, expected {self.target_object}",
                "yellow",
            )
            return

        frame_info = self._load_frame_info(attrs, demo_dir)
        self._process_frames(frame_info, demo_dir)
        self.episode_ends_arrays.append(self.total_count)

    def _load_attributes(self, dir: Path):
        attr_fpath = dir / "attributes.json"
        attr_pkl_fpath = dir / "attributes.pkl"
        if attr_fpath.exists():
            with open(attr_fpath, "r") as f:
                attrs = json.load(f)
        elif attr_pkl_fpath.exists():
            with open(attr_pkl_fpath, "rb") as f:
                attrs = pickle.load(f)
        else:
            raise RuntimeError("Invalid dataset format!")
        return attrs

    def _load_frame_info(self, attrs: dict, demo_dir: Path) -> list:
        if "frame_info" in attrs:
            return attrs["frame_info"]
        else:
            frame_info_path = demo_dir.joinpath("frame_info.json")
            with open(frame_info_path, "r") as f:
                return json.load(f)

    def _read_img(self, imgs_dir: Path, idx: int, which_cam: str, depth: bool = False):
        if not depth:
            img_path = imgs_dir.joinpath(f"{idx}_{which_cam}.png")
            img = Image.open(img_path).convert("RGB")
        else:
            img_path = imgs_dir.joinpath(f"{idx}_{which_cam}_registered.png")
            img = Image.open(img_path)
        img = np.array(img)

        if not depth:
            img = preproces_rgb_image(img, which_cam, self.img_size)
        else:
            img = preproces_image(np.expand_dims(img, axis=-1), self.img_size)
            img = img.squeeze(-1)

        return img

    def _load_pointcloud(self, pcd_dir: Path, idx: int, which_cam: str):
        pcd_path = pcd_dir.joinpath(f"{idx}_{which_cam}_processed.npy")
        pcdz_path = pcd_dir.joinpath(f"{idx}_{which_cam}_processed.npz")

        if pcdz_path.exists():
            pcd = np.load(pcdz_path)["pcd"]
        elif pcd_path.exists():
            pcd = np.load(pcd_path)
        else:
            raise RuntimeError(f"Pointcloud for idx '{idx}' is not found")

        pcd = preprocess_point_cloud(pcd, self.num_points, which_cam, use_cuda=True)
        return pcd

    def _process_frames(self, frame_info: list, demo_dir: Path):
        imgs_dir = demo_dir.joinpath("rgb")
        depth_dir = demo_dir.joinpath("depth")
        pcd_dir = demo_dir.joinpath("point_cloud")

        demo_length = len(frame_info)
        for i in tqdm.tqdm(range(demo_length)):
            self.total_count += 1

            info = frame_info[i]
            idx = info["frame_id"]

            if self.which_cam == "both":
                self.img_left_batch_buffer.append(self._read_img(imgs_dir, idx, "left"))
                self.img_right_batch_buffer.append(
                    self._read_img(imgs_dir, idx, "right")
                )
                self.depth_left_batch_buffer.append(
                    self._read_img(depth_dir, idx, "left", depth=True)
                )
                self.depth_right_batch_buffer.append(
                    self._read_img(depth_dir, idx, "right", depth=True)
                )
                self.pc_left_batch_buffer.append(
                    self._load_pointcloud(pcd_dir, idx, "left")
                )
                self.pc_right_batch_buffer.append(
                    self._load_pointcloud(pcd_dir, idx, "right")
                )
            else:
                self.img_batch_buffer.append(
                    self._read_img(imgs_dir, idx, self.which_cam)
                )
                self.depth_batch_buffer.append(
                    self._read_img(depth_dir, idx, self.which_cam, depth=True)
                )
                self.pc_batch_buffer.append(
                    self._load_pointcloud(pcd_dir, idx, self.which_cam)
                )

            left_eef = np.array(info["robot_states"]["left_eef"]).reshape(4, 4)
            right_eef = np.array(info["robot_states"]["right_eef"]).reshape(4, 4)

            if i < (demo_length - 1):
                next_info = frame_info[i + 1]
                next_left_eef = next_info["robot_states"]["left_eef"]
                next_pos_left = np.array(next_left_eef).reshape(4, 4)
                next_right_eef = next_info["robot_states"]["right_eef"]
                next_pos_right = np.array(next_right_eef).reshape(4, 4)
            else:
                next_pos_left = None
                next_pos_right = None

            action = np.zeros(16, dtype=np.float32)
            action[:7] = get_pose_delta(left_eef, next_pos_left)
            action[7] = info["robot_states"]["left_gripper"]
            action[8:15] = get_pose_delta(right_eef, next_pos_right)
            action[15] = info["robot_states"]["right_gripper"]

            robot_state = np.zeros(16, dtype=np.float32)
            robot_state[:7] = get_pose_vec(left_eef)
            robot_state[7] = info["robot_states"]["left_gripper"]
            robot_state[8:15] = get_pose_vec(right_eef)
            robot_state[15] = info["robot_states"]["right_gripper"]

            self.action_batch_buffer.append(action)
            self.state_batch_buffer.append(robot_state)

            if len(self.action_batch_buffer) >= self.chunk_size:
                self._flush_buffers_to_zarr()

    def finalize_dataset(self):
        self._flush_buffers_to_zarr()  # Flush any remaining data

        episode_ends_np_array = np.array(self.episode_ends_arrays)
        compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
        self.zarr_meta.create_dataset(
            "episode_ends",
            data=episode_ends_np_array,
            chunks=(100,),
            dtype="int64",
            overwrite=True,
            compressor=compressor,
        )

        if self.which_cam == "both":
            cprint(
                f"left_img shape: {self.z_img_left.shape}, range: [{np.min(self.z_img_left)}, {np.max(self.z_img_left)}]",
                "green",
            )
            cprint(
                f"right_img shape: {self.z_img_right.shape}, range: [{np.min(self.z_img_right)}, {np.max(self.z_img_right)}]",
                "green",
            )
            cprint(
                f"left_depth shape: {self.z_depth_left.shape}, range: [{np.min(self.z_depth_left)}, {np.max(self.z_depth_left)}]",
                "green",
            )
            cprint(
                f"right_depth shape: {self.z_depth_right.shape}, range: [{np.min(self.z_depth_right)}, {np.max(self.z_depth_right)}]",
                "green",
            )
            cprint(
                f"left_pcd shape: {self.z_pc_left.shape}, range: [{np.min(self.z_pc_left)}, {np.max(self.z_pc_left)}]",
                "green",
            )
            cprint(
                f"right_pcd shape: {self.z_pc_right.shape}, range: [{np.min(self.z_pc_right)}, {np.max(self.z_pc_right)}]",
                "green",
            )
        else:
            cprint(
                f"img shape: {self.z_img.shape}, range: [{np.min(self.z_img)}, {np.max(self.z_img)}]",
                "green",
            )
            cprint(
                f"point_cloud shape: {self.z_pc.shape}, range: [{np.min(self.z_pc)}, {np.max(self.z_pc)}]",
                "green",
            )
            cprint(
                f"depth shape: {self.z_depth.shape}, range: [{np.min(self.z_depth)}, {np.max(self.z_depth)}]",
                "green",
            )
        cprint(
            f"action shape: {self.z_action.shape}, range: [{np.min(self.z_action)}, {np.max(self.z_action)}]",
            "green",
        )
        cprint(
            f"state shape: {self.z_state.shape}, range: [{np.min(self.z_state)}, {np.max(self.z_state)}]",
            "green",
        )
        cprint(
            f"episode_ends shape: {episode_ends_np_array.shape}, range: [{np.min(episode_ends_np_array)}, {np.max(episode_ends_np_array)}]",
            "green",
        )
        cprint(f"total_count: {self.total_count}", "green")
        cprint(f"Saved zarr file to {self.save_data_path}", "green")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert real robot demonstration data to zarr format."
    )
    parser.add_argument(
        "input_dir", type=str, help="Path to the directory containing expert demos."
    )
    parser.add_argument(
        "output_dir", type=str, help="Path to save the output zarr dataset."
    )
    parser.add_argument(
        "--num-demo", "-n",
        type=int,
        help="Set the number of episodes to include in dataset",
    )
    parser.add_argument(
        "--which_cam", "-c",
        type=str,
        default="both",
        choices=["left", "right", "both"],
        help="Which camera to process.",
    )
    parser.add_argument(
        "--num_points", "-p",
        type=int,
        default=4096,
        help="Number of points to sample in the point cloud.",
    )
    parser.add_argument(
        "--img_size", type=int, default=224, help="Size to resize RGB and depth images."
    )
    parser.add_argument(
        "--target_object",
        type=str,
        default="tomato",
        help="Target object to filter demonstrations.",
    )
    # parser.add_argument(
    #     "--overwrite-output", action="store_true", help="Overwrite existing output."
    # )
    args = parser.parse_args()

    CHUNK_SIZE = 100

    expert_data_path = Path(args.input_dir)
    demo_dirs = [d for d in expert_data_path.iterdir() if d.is_dir()]
    if args.num_demo is not None:
        num_demo = min(args.num_demo, len(demo_dirs))
        demo_dirs.sort()
        demo_dirs = demo_dirs[: args.num_demo]

    cprint(f"Dataset directory: {expert_data_path}", "green")

    data_name = f"{expert_data_path.name}_{len(demo_dirs)}demo_{args.which_cam}_{args.num_points}.zarr"
    save_data_path = Path(args.output_dir).joinpath(data_name)
    cprint(f"Output directory: {save_data_path}", "green")

    overwrite_output = False
    if save_data_path.exists():
        cprint("Data already exists at {}".format(save_data_path), "red")
        user_input = input("\033[91mDo you want to overwrite? ([y]/n)\033[0m: ") or "y"
        if user_input.lower() == "y":
            overwrite_output = True
        else:
            cprint("Exiting", "red")
            exit()

    converter = DataConverter(
        target_object=args.target_object,
        which_cam=args.which_cam,
        save_data_path=save_data_path,
        overwrite_output=overwrite_output,
        chunk_size=CHUNK_SIZE,
        num_points=args.num_points,
        img_size=args.img_size,
    )

    for idx, demo_dir in enumerate(demo_dirs):
        cprint(f"[{idx+1}/{len(demo_dirs)}] Processing {demo_dir.name}", "green")
        converter.process_demo_directory(demo_dir)

    converter.finalize_dataset()
