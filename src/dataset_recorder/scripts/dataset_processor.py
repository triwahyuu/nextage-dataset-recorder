import numpy as np
import cv2
import time
import json
import logging
import pickle
import torch

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
from tqdm import tqdm
from scipy.spatial.transform import Rotation
from cv_bridge import CvBridge


# Logger setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


@dataclass
class CameraParameters:
    """Dataclass to hold camera intrinsic and distortion parameters."""

    K: np.ndarray  # 3x3 intrinsic matrix K
    D: np.ndarray  # Distortion coefficients D
    image_size: Tuple[int, int]  # (width, height)


@dataclass
class CameraInfo:
    rgb_cam: CameraParameters
    depth_cam: CameraParameters
    depth2rgb: np.ndarray
    rgb2world: np.ndarray
    rgb_map1: np.ndarray = None
    rgb_map2: np.ndarray = None
    rgb_k_new: np.ndarray = None
    depth_map1: np.ndarray = None
    depth_map2: np.ndarray = None
    depth_k_new: np.ndarray = None


class DatasetClipProcessor:
    """
    Dataset postprocessing pipeline for generating point clouds from RGB-Depth image pairs.
    Handles image rectification, depth registration, and point cloud generation.
    """

    def __init__(
        self,
        clip_dir: Path,
        no_pointcloud: bool = False,
        depth_scale: float = 1000.0,
        device=torch.device("cuda"),
    ):
        """
        Initialize the pipeline.

        Args:
            clip_dir (Path): clip directory.
            depth_scale (float): Scale factor to convert depth values to meters.
            use_gpu (bool): Whether to use GPU if available.
        """
        self.clip_dir = Path(clip_dir).resolve()
        self.no_pointcloud = no_pointcloud

        self.logger = logging.getLogger(__name__)

        self.device = device
        self.is_using_gpu = self.device == torch.device("cuda")

        self.attributes = self._load_attributes(self.clip_dir)
        if "frame_info" in self.attributes:
            self.frame_info = self.attributes["frame_info"]
        else:
            self.frame_info = self._load_frame_info(self.clip_dir)
        self.depth_scale = depth_scale
        self.msg_idx_map = self.attributes["attributes"]["message_idx_map"]

        self.bridge = CvBridge()

        self.camera_info = {
            "left": self._get_camera_info(self.attributes, "left"),
            "right": self._get_camera_info(self.attributes, "right"),
        }

        self.rgb_dir = self.clip_dir / "rgb"
        self.depth_dir = self.clip_dir / "depth"
        self.pcd_dir = self.clip_dir / "point_cloud"

    def _get_camera_info(self, attrs: dict, which_cam: str):
        rgb_cam = self._load_camera_info(attrs, which_cam, "rgb")
        depth_cam = self._load_camera_info(attrs, which_cam, "depth")
        depth2rgb = self._get_depth2rgb(attrs, which_cam)
        rgb2world = self._get_rgb2world(attrs, which_cam)

        caminfo = CameraInfo(
            rgb_cam=rgb_cam,
            depth_cam=depth_cam,
            depth2rgb=depth2rgb,
            rgb2world=rgb2world,
        )
        self._compute_rectification_maps(caminfo)
        return caminfo

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

    def _load_frame_info(self, dir: Path):
        attr_fpath = dir / "frame_info.json"
        with open(attr_fpath, "r") as f:
            attrs = json.load(f)
        return attrs

    def _load_camera_info(self, attributes: dict, which_cam: str, cam_type: str):
        all_cam_attrs = attributes["attributes"]["camera"][which_cam]
        cam_attr = all_cam_attrs["cam_info"][cam_type]

        K = np.array(cam_attr["K"]).reshape(3, 3)
        D = np.array(cam_attr["D"])
        if self.is_using_gpu:
            K = torch.from_numpy(K).to(self.device)
            D = torch.from_numpy(D).to(self.device)
        return CameraParameters(K, D, (cam_attr["width"], cam_attr["height"]))

    def _get_tf_mat(self, attrs: dict, which_cam: str, tf_name: str):
        cam_pose = attrs["attributes"]["camera"][which_cam]["cam_pose"]
        if tf_name not in cam_pose:
            return None

        rot = cam_pose[tf_name]["rotation"]
        rot_np = np.array([rot["x"], rot["y"], rot["z"], rot["w"]])
        trans = cam_pose[tf_name]["translation"]
        trans_np = np.array([trans["x"], trans["y"], trans["z"]])

        tf_mat = np.eye(4, dtype=np.float32)
        tf_mat[:3, :3] = Rotation.from_quat(rot_np).as_matrix()
        tf_mat[:3, 3] = np.array(trans_np)
        if self.is_using_gpu:
            tf_mat = torch.from_numpy(tf_mat).to(self.device)
        return tf_mat

    def _get_depth2rgb(self, attrs: dict, which_cam: str):
        cam_pose = attrs["attributes"]["camera"][which_cam]["cam_pose"]
        if "depth2rgb" not in cam_pose:
            return self._calculate_depth2rgb(attrs, which_cam)

        return self._get_tf_mat(attrs, which_cam, "depth2rgb")

    def _get_rgb2world(self, attrs: dict, which_cam: str):
        cam_pose = attrs["attributes"]["camera"][which_cam]["cam_pose"]
        if "rgb2world" not in cam_pose:
            return self._calculate_rgb2world(attrs, which_cam)

        return self._get_tf_mat(attrs, which_cam, "rgb2world")

    def _calculate_depth2rgb(self, attrs: dict, which_cam: str):
        depth2cam = self._get_tf_mat(attrs, which_cam, "depth2camera")
        rgb2cam = self._get_tf_mat(attrs, which_cam, "rgb2camera")

        if self.is_using_gpu:
            depth2rgb = torch.linalg.inv(rgb2cam) @ depth2cam
        else:
            depth2rgb = np.linalg.inv(rgb2cam) @ depth2cam
        return depth2rgb

    def _calculate_rgb2world(self, attrs: dict, which_cam: str):
        cam2world = self._get_tf_mat(attrs, which_cam, "camera2world")
        rgb2cam = self._get_tf_mat(attrs, which_cam, "rgb2camera")

        rgb2world = cam2world @ rgb2cam
        return rgb2world

    def _compute_rectification_maps(self, cam_info: CameraInfo):
        """Pre-compute rectification maps for faster processing."""
        rgb_imgsz = cam_info.rgb_cam.image_size
        d_img_size = cam_info.depth_cam.image_size

        rgb_k = cam_info.rgb_cam.K
        rgb_d = cam_info.rgb_cam.D
        depth_k = cam_info.depth_cam.K
        depth_d = cam_info.depth_cam.D
        if self.is_using_gpu:
            rgb_k = rgb_k.cpu().numpy()
            rgb_d = rgb_d.cpu().numpy()
            depth_k = depth_k.cpu().numpy()
            depth_d = depth_d.cpu().numpy()

        # Compute optimal camera matrix for undistortion
        rgb_K_new, _ = cv2.getOptimalNewCameraMatrix(
            rgb_k, rgb_d, rgb_imgsz, 1, rgb_imgsz
        )
        depth_K_new, _ = cv2.getOptimalNewCameraMatrix(
            depth_k, depth_d, d_img_size, 1, rgb_imgsz
        )

        # Pre-compute rectification maps
        cam_info.rgb_map1, cam_info.rgb_map2 = cv2.initUndistortRectifyMap(
            rgb_k, rgb_d, None, rgb_K_new, rgb_imgsz, cv2.CV_16SC2
        )
        cam_info.depth_map1, cam_info.depth_map2 = cv2.initUndistortRectifyMap(
            depth_k, depth_d, None, depth_K_new, rgb_imgsz, cv2.CV_16SC2
        )

        # Update camera matrices
        if self.is_using_gpu:
            rgb_K_new = torch.from_numpy(rgb_K_new.astype(np.float32)).to(self.device)
            depth_K_new = torch.from_numpy(depth_K_new.astype(np.float32)).to(
                self.device
            )
        cam_info.rgb_k_new = rgb_K_new
        cam_info.depth_k_new = depth_K_new

    def rectify_images(
        self, rgb_image: np.ndarray, depth_image: np.ndarray, cam_info: CameraInfo
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rectify RGB and depth images to remove distortion.

        Args:
            rgb_image: Input RGB image
            depth_image: Input depth image

        Returns:
            Tuple of (rectified_rgb, rectified_depth)
        """
        h, w = rgb_image.shape[:2]
        assert cam_info.rgb_cam.image_size == (w, h)
        dh, dw = depth_image.shape[:2]
        assert cam_info.depth_cam.image_size == (dw, dh)

        # Vectorized rectification using pre-computed maps
        rgb_rectified = cv2.remap(
            rgb_image, cam_info.rgb_map1, cam_info.rgb_map2, cv2.INTER_LINEAR
        )
        depth_rectified = cv2.remap(
            depth_image, cam_info.depth_map1, cam_info.depth_map2, cv2.INTER_NEAREST
        )
        # depth_rectified = depth_image.astype(np.float32)

        return rgb_rectified, depth_rectified

    def process_image_pair(self, msgs: dict, which_cam: str, frame_id: str) -> str:
        """
        Process a single RGB-Depth image pair and generate point cloud from info.

        Args:
            info (dict): rgbd image pair info

        Returns:
            Path to saved point cloud file
        """
        out_pcd_path = self.pcd_dir / f"{frame_id}_{which_cam}_processed.npy"
        if out_pcd_path.exists():
            return out_pcd_path

        cam_info = self.camera_info[which_cam]
        rgb_image, depth_image = self._get_rgbd_pair(msgs, which_cam)
        self._save_rgbd_pair(rgb_image, depth_image, which_cam, frame_id)

        if self.no_pointcloud:
            return

        # 1. Rectify images
        rgb_rect, depth_rect = self.rectify_images(rgb_image, depth_image, cam_info)

        rgb_height, rgb_width = rgb_rect.shape[:2]
        depth_height, depth_width = depth_rect.shape[:2]
        assert (rgb_height, rgb_width) == (depth_height, depth_width)

        # 2. Prepare depth data (convert to meters)
        depth_m = depth_rect.astype(np.float32) / self.depth_scale

        # 3. Create pixel coordinate grid for the depth image
        u_grid, v_grid = np.meshgrid(np.arange(depth_width), np.arange(depth_height))

        u_flat = u_grid.flatten()
        v_flat = v_grid.flatten()
        z_flat = depth_m.flatten()

        # 4. Filter out invalid depth values
        valid_depth_mask = (z_flat > 1e-6) & np.isfinite(z_flat)

        u_valid = u_flat[valid_depth_mask]
        v_valid = v_flat[valid_depth_mask]
        z_valid = z_flat[valid_depth_mask]

        if z_valid.size == 0:
            self.logger.warning(
                f"[{frame_id}_{which_cam}] No valid depth points found after initial filtering."
            )
            return

        # 5. Unproject valid depth pixels to 3D points in the depth camera's coordinate frame
        depth_k = cam_info.depth_k_new
        x_valid = (u_valid - depth_k[0, 2]) * z_valid / depth_k[0, 0]
        y_valid = (v_valid - depth_k[1, 2]) * z_valid / depth_k[1, 1]

        pts_valid = np.vstack((x_valid, y_valid, z_valid))  # Shape: (3, N)

        # 6. Transform points from depth camera frame to RGB camera frame
        d2r_t = cam_info.depth2rgb[:3, 3].reshape(3, 1)
        d2r_r = cam_info.depth2rgb[:3, :3]

        # Shape: (3, N)
        pts_rgb_tf: np.ndarray = np.dot(d2r_r, pts_valid) + d2r_t

        # 7. Filter points that are behind or too close to the RGB camera plane
        z_rgb_tf = pts_rgb_tf[2, :]
        front_cam_mask = z_rgb_tf > 1e-6

        # Points in RGB camera frame that are in front of the camera
        # Shape: (3, N_front)
        pts_rgb_front = pts_rgb_tf[:, front_cam_mask]

        if pts_rgb_front.shape[1] == 0:
            self.logger.warning(
                f"[{frame_id}_{which_cam}] No points in front of RGB camera after transformation."
            )
            return
        self.logger.debug(
            f"{pts_rgb_front.shape[1]} points remaining after filtering behind-camera points."
        )

        # Extract X, Y, Z coordinates for projection
        x_front = pts_rgb_front[0, :]
        y_front = pts_rgb_front[1, :]
        z_front = pts_rgb_front[2, :]  # Z-depths in RGB camera frame

        # 8. Project 3D points (in RGB camera frame) onto the RGB image plane (float coordinates)
        # Denominator z_front should not be zero here due to "front_cam_mask" (z_rgb_tf > 1e-6)
        rgb_k = cam_info.rgb_k_new
        u_proj_f = (rgb_k[0, 0] * x_front / z_front) + rgb_k[0, 2]
        v_proj_f = (rgb_k[1, 1] * y_front / z_front) + rgb_k[1, 2]

        # 9. Round projected coordinates to integer indices and filter points that fall outside RGB image bounds AFTER rounding
        u_proj_idx = np.round(u_proj_f).astype(int)
        v_proj_idx = np.round(v_proj_f).astype(int)

        # Create mask for valid integer indices (must be >= 0 and < dimension)
        valid_u_mask = (u_proj_idx >= 0) & (u_proj_idx < rgb_width)
        valid_v_mask = (v_proj_idx >= 0) & (v_proj_idx < rgb_height)
        in_bounds_mask = valid_u_mask & valid_v_mask

        u_img_idx = u_proj_idx[in_bounds_mask]
        v_img_idx = v_proj_idx[in_bounds_mask]

        pts_final = pts_rgb_front[:, in_bounds_mask]  # (3, N_final)
        z_final = pts_final[2, :]  # Shape: (N_final,)

        # Check if any points remain after all filtering
        if pts_final.shape[1] == 0:
            self.logger.warning(
                f"[{frame_id}_{which_cam}] No points projected within RGB image bounds "
                "after rounding and final filtering."
            )
            return
        self.logger.debug(f"{pts_final.shape[1]} points remaining.")

        # 10. Get colors for these final points from the RGB image
        # This line should now be safe due to the refined filtering in step 9
        colors_bgr = rgb_rect[v_img_idx, u_img_idx]  # (N_final, 3)
        colors_rgb = colors_bgr[:, ::-1]  # Convert BGR to RGB

        # 11. Transform points to world frame
        pts_final_homogen = np.vstack((pts_final, np.ones((1, pts_final.shape[1]))))
        pts_world_homogen = cam_info.rgb2world @ pts_final_homogen
        pts_final = pts_world_homogen[:3, :]

        # 12. Create and save the registered depth image
        reg_depth_m = np.zeros((rgb_height, rgb_width), dtype=np.float32)
        reg_depth_m[v_img_idx, u_img_idx] = z_final
        reg_depth = (reg_depth_m * self.depth_scale).astype(np.uint16)

        point_cloud = np.hstack((pts_final.T, colors_rgb))  # Shape: (N, 6)
        if point_cloud.shape[0] < rgb_height * rgb_width * 0.04:
            self.logger.warning(
                f"[{frame_id}_{which_cam}] Generated point cloud is too small with {point_cloud.shape[0]} points"
            )

        return self._save_depthreg_pcd(point_cloud, reg_depth, which_cam, frame_id)

    def process_image_pair_pytorch(
        self, msgs: dict, which_cam: str, frame_id: str
    ) -> str:
        """
        Process a single RGB-Depth image pair using PyTorch and generate point cloud.
        Image rectification is done by OpenCV (CPU), rest is on self.device (GPU if available).
        """
        out_pcd_path = self.pcd_dir / f"{frame_id}_{which_cam}_processed.npy"
        if out_pcd_path.exists():
            return out_pcd_path

        cam_info = self.camera_info[which_cam]
        rgb_image_np, depth_image_np = self._get_rgbd_pair(msgs, which_cam)
        self._save_rgbd_pair(rgb_image_np, depth_image_np, which_cam, frame_id)

        if self.no_pointcloud:
            return

        # 1. Rectify images
        rgb_rect_np, depth_rect_np = self.rectify_images(
            rgb_image_np, depth_image_np, cam_info
        )

        rgb_height, rgb_width = rgb_rect_np.shape[:2]
        depth_height, depth_width = depth_rect_np.shape[:2]
        assert (rgb_height, rgb_width) == (depth_height, depth_width)

        rgb_rect = torch.from_numpy(rgb_rect_np.astype(np.float32)).to(self.device)
        depth_rect = torch.from_numpy(depth_rect_np.astype(np.float32)).to(self.device)

        # 2. Prepare depth data (convert to meters)
        depth_m = depth_rect / self.depth_scale

        # 3. Create pixel coordinate grid for the depth image
        u_coords = torch.arange(depth_width, device=self.device, dtype=torch.float32)
        v_coords = torch.arange(depth_height, device=self.device, dtype=torch.float32)
        u_grid, v_grid = torch.meshgrid(u_coords, v_coords, indexing="xy")

        u_flat = u_grid.flatten()
        v_flat = v_grid.flatten()
        z_flat = depth_m.flatten()

        # 4. Filter out invalid depth values
        valid_depth_mask = (z_flat > 1e-6) & torch.isfinite(z_flat)

        u_valid = u_flat[valid_depth_mask]
        v_valid = v_flat[valid_depth_mask]
        z_valid = z_flat[valid_depth_mask]

        if z_valid.numel() == 0:
            self.logger.warning(
                f"[{frame_id}_{which_cam}] No valid depth points after initial filtering."
            )
            return None

        # 5. Unproject valid depth pixels to 3D points in the depth camera's coordinate frame
        depth_k = cam_info.depth_k_new
        x_valid = (u_valid - depth_k[0, 2]) * z_valid / depth_k[0, 0]
        y_valid = (v_valid - depth_k[1, 2]) * z_valid / depth_k[1, 1]

        # pts_valid shape: (3, N_valid)
        pts_valid = torch.stack((x_valid, y_valid, z_valid), dim=0)

        # 6. Transform points from depth camera frame to RGB camera frame
        # pts_rgb_tf shape: (3, N_valid)
        d2r_t = cam_info.depth2rgb[:3, 3].reshape(3, 1)
        d2r_r = cam_info.depth2rgb[:3, :3]
        pts_rgb_tf = torch.matmul(d2r_r, pts_valid) + d2r_t

        # 7. Filter points that are behind or too close to the RGB camera plane
        z_rgb_tf = pts_rgb_tf[2, :]
        front_cam_mask = z_rgb_tf > 1e-6

        # pts_rgb_front shape: (3, N_front)
        pts_rgb_front = pts_rgb_tf[:, front_cam_mask]

        if pts_rgb_front.shape[1] == 0:
            self.logger.warning(
                f"[{frame_id}_{which_cam}] No points in front of RGB camera after transformation."
            )
            return None
        self.logger.debug(
            f"{pts_rgb_front.shape[1]} points remaining after filtering behind-camera points."
        )

        x_front = pts_rgb_front[0, :]
        y_front = pts_rgb_front[1, :]
        z_front = pts_rgb_front[2, :]  # Z-depths in RGB camera frame

        # 8. Project 3D points (in RGB camera frame) onto the RGB image plane
        rgb_k = cam_info.rgb_k_new
        u_proj_f = (rgb_k[0, 0] * x_front / z_front) + rgb_k[0, 2]
        v_proj_f = (rgb_k[1, 1] * y_front / z_front) + rgb_k[1, 2]

        # 9. Round projected coordinates and filter out-of-bounds
        u_proj_idx = torch.round(u_proj_f).long()
        v_proj_idx = torch.round(v_proj_f).long()

        valid_u_mask = (u_proj_idx >= 0) & (u_proj_idx < rgb_width)
        valid_v_mask = (v_proj_idx >= 0) & (v_proj_idx < rgb_height)
        in_bounds_mask = valid_u_mask & valid_v_mask

        u_img_idx = u_proj_idx[in_bounds_mask]
        v_img_idx = v_proj_idx[in_bounds_mask]

        pts_final = pts_rgb_front[:, in_bounds_mask]  # shape: (3, N_final)
        z_final = pts_final[2, :]  # shape: (N_final,)

        if pts_final.shape[1] == 0:
            self.logger.warning(
                f"[{frame_id}_{which_cam}] (PyTorch) No points projected within RGB image bounds."
            )
            return None

        # 10. Get colors for these final points from the RGB image
        colors_bgr = rgb_rect[v_img_idx, u_img_idx]  # (N_final, 3)
        colors_rgb = colors_bgr[:, [2, 1, 0]]  # Convert BGR to RGB

        # 11. Transform points to world frame
        # pts_final_homogen shape: (4, N_final)
        ones_tensor = torch.ones(
            (1, pts_final.shape[1]), device=self.device, dtype=torch.float32
        )
        pts_final_homogen = torch.cat((pts_final, ones_tensor), dim=0)
        pts_world_homogen = torch.matmul(cam_info.rgb2world, pts_final_homogen)
        pts_final_world = pts_world_homogen[:3, :]  # shape: (3, N_final)

        # 12. Create and save the registered depth image and point cloud
        # Registered depth image
        reg_depth_m = torch.zeros(
            (rgb_height, rgb_width), dtype=torch.float32, device=self.device
        )
        reg_depth_m[v_img_idx, u_img_idx] = z_final

        reg_depth_scaled = reg_depth_m * self.depth_scale
        reg_depth = reg_depth_scaled.cpu().numpy().astype(np.uint16)

        # Point cloud
        point_cloud = torch.cat((pts_final_world.T, colors_rgb.float()), dim=1)
        point_cloud_np: np.ndarray = point_cloud.cpu().numpy()

        if point_cloud_np.shape[0] < rgb_height * rgb_width * 0.04:
            self.logger.warning(
                f"[{frame_id}_{which_cam}] Generated point cloud is small: {point_cloud_np.shape[0]} points."
            )

        return self._save_depthreg_pcd(point_cloud_np, reg_depth, which_cam, frame_id)

    def _save_rgbd_pair(self, rgb_image, depth_image, which_cam, frame_id):
        self.rgb_dir.mkdir(parents=True, exist_ok=True)
        rgb_path = self.rgb_dir / f"{frame_id}_{which_cam}.png"
        cv2.imwrite(str(rgb_path), rgb_image)

        self.depth_dir.mkdir(parents=True, exist_ok=True)
        depth_path = self.depth_dir / f"{frame_id}_{which_cam}.png"
        cv2.imwrite(str(depth_path), depth_image)

    def _save_depthreg_pcd(
        self, pcd: np.ndarray, reg_depth: np.ndarray, which_cam: str, frame_id: str
    ):
        self.pcd_dir.mkdir(parents=True, exist_ok=True)
        out_reg_depth_path = self.depth_dir / f"{frame_id}_{which_cam}_registered.png"
        cv2.imwrite(str(out_reg_depth_path), reg_depth)

        # out_pcd_path = self.pcd_dir / f"{frame_id}_{which_cam}_processed.npy"
        # np.save(out_pcd_path, pcd)
        out_pcd_path = self.pcd_dir / f"{frame_id}_{which_cam}_processed.npz"
        np.savez_compressed(out_pcd_path, pcd=pcd)
        return str(out_pcd_path)

    def _get_rgbd_pair(
        self, msgs: dict, which_cam: str
    ) -> Tuple[np.ndarray, np.ndarray]:

        ns = f"kinect_{which_cam}"

        rgb_msg = msgs[self.msg_idx_map[f"{ns}/image"]]
        rgb_img = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")

        depth_msg = msgs[self.msg_idx_map[f"{ns}/depth"]]
        depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")

        return rgb_img, depth_img

    def process_frame(self, msgs: dict, frame_id: str):
        self.logger.debug(f"Processing frame {frame_id}")
        if self.is_using_gpu:
            outpath_left = self.process_image_pair_pytorch(msgs, "left", frame_id)
            outpath_right = self.process_image_pair_pytorch(msgs, "right", frame_id)
        else:
            outpath_left = self.process_image_pair(msgs, "left", frame_id)
            outpath_right = self.process_image_pair(msgs, "right", frame_id)
        return {"left": outpath_left, "right": outpath_right}

    def run(self) -> list:
        """
        Run processing on RGB-Depth image pairs for the clip.
        """

        output_files = []
        total_start_time = time.time()

        for info in tqdm(self.frame_info):
            frame_id = info["frame_id"]

            left_pcd_path = self.pcd_dir / f"{frame_id}_left_processed.npy"
            right_pcd_path = self.pcd_dir / f"{frame_id}_right_processed.npy"
            if left_pcd_path.exists() and right_pcd_path.exists():
                outpath = {"left": left_pcd_path, "right": right_pcd_path}
                output_files.append(outpath)
                continue

            left_pcd_path = self.pcd_dir / f"{frame_id}_left_processed.npz"
            right_pcd_path = self.pcd_dir / f"{frame_id}_right_processed.npz"
            if left_pcd_path.exists() and right_pcd_path.exists():
                outpath = {"left": left_pcd_path, "right": right_pcd_path}
                output_files.append(outpath)
                continue

            msg_path = self.clip_dir / f"{info['messages_path']}"
            with open(msg_path, "rb") as f:
                msgs = pickle.load(f)

            outpath = self.process_frame(msgs, frame_id)
            output_files.append(outpath)

        total_time = (time.time() - total_start_time) * 1000
        self.logger.debug(
            f"Processed {len(output_files)} image pairs in {total_time/1000:.2f}s"
        )
        self.logger.debug(
            f"Average time per pair: {total_time/len(output_files):.2f} ms"
        )

        return output_files


class DatasetProcessor:
    def __init__(
        self, dataset_dir, no_pointcloud=False, depth_scale=1000.0, use_gpu=True
    ):
        self.dataset_dir = Path(dataset_dir).resolve()
        self.depth_scale = float(depth_scale)

        self.no_pointcloud = no_pointcloud

        self.logger = logging.getLogger(__name__)

        # Initialize device
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.logger.info(f"Using device: {self.device}")
        if self.no_pointcloud:
            self.logger.info(f"Not generating point cloud")

        self.clip_dirs = [d for d in self.dataset_dir.iterdir() if d.is_dir()]

        if len(self.clip_dirs) == 0:
            raise RuntimeError(f"Dataset in {self.dataset_dir} is empty.")

    def run(self):
        num_clips = len(self.clip_dirs)
        for idx, clip_dir in enumerate(self.clip_dirs):
            self.logger.info(f"[{idx}/{num_clips}] Processing clip {clip_dir}")
            processor = DatasetClipProcessor(
                clip_dir, self.no_pointcloud, self.depth_scale, device=self.device
            )
            processor.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process dataset.")
    parser.add_argument(
        "dataset_dir",
        nargs="?",
        type=str,
        help="Path to dataset directory",
        default="/workspaces/dataset_recorder/recordings",
    )
    parser.add_argument(
        "--no-pointcloud", action="store_true", help="Do not generate pointcloud."
    )

    args = parser.parse_args()

    processor = DatasetProcessor(args.dataset_dir, args.no_pointcloud)
    processor.run()
