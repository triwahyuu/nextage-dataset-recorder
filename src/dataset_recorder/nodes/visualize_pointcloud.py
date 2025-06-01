#!/usr/bin/env python

import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from pathlib import Path


class PointCloudPublisher:
    def __init__(self):
        rospy.init_node("pointcloud_publisher", anonymous=True)

        # Get the .npy file path from command line argument or parameter
        npy_file_path = rospy.get_param("~pointcloud_file", "pointcloud.npy")
        self.npy_file_path = Path(npy_file_path).resolve()

        # Publisher with larger queue and latching for persistence
        self.pub = rospy.Publisher(
            "/pointcloud", PointCloud2, queue_size=10, latch=True
        )

        # Frame ID for the point cloud
        self.frame_id = rospy.get_param("~frame_id", "map")

        # Pre-define PointCloud2 fields to avoid repeated creation
        self.fields = [
            PointField("x", 0, PointField.FLOAT32, 1),
            PointField("y", 4, PointField.FLOAT32, 1),
            PointField("z", 8, PointField.FLOAT32, 1),
            PointField("rgb", 12, PointField.UINT32, 1),
        ]

        # Load and publish the point cloud
        self.load_and_publish_pointcloud()

    def load_and_publish_pointcloud(self):
        """Load point cloud from .npy file and publish it"""
        try:
            # Load the numpy array
            rospy.logdebug(f"Loading point cloud from: {self.npy_file_path}")

            if not self.npy_file_path.exists():
                rospy.logerr(f"File not found: {self.npy_file_path}")
                return

            # Load with memory mapping for large files (faster for big datasets)
            pc_data = np.load(self.npy_file_path, mmap_mode="r")

            if pc_data.shape[1] != 6:
                rospy.logerr(
                    f"Expected 6 columns (x,y,z,r,g,b), got {pc_data.shape[1]}"
                )
                return

            rospy.logdebug(f"Loaded point cloud with {pc_data.shape[0]} points")

            # Create PointCloud2 message with optimized method
            start_time = rospy.Time.now()
            pc_msg = self.create_pointcloud2_vectorized(pc_data)
            creation_time = (rospy.Time.now() - start_time).to_sec()

            rospy.logdebug(
                f"Point cloud message created in {creation_time:.3f} seconds"
            )

            # Publish the point cloud
            self.pub.publish(pc_msg)
            rospy.loginfo("Point cloud published successfully!")

        except Exception as e:
            rospy.logerr(f"Error loading point cloud: {str(e)}")

    def create_pointcloud2_vectorized(self, points):
        """
        Create a PointCloud2 message from numpy array using vectorized operations
        points: numpy array of shape (N, 6) with columns [x, y, z, r, g, b]
        """
        n_points = points.shape[0]

        # Create header
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = self.frame_id

        # Vectorized RGB packing - much faster than loop
        # Ensure RGB values are uint8 and in valid range
        rgb_data = np.clip(points[:, 3:6], 0, 255).astype(np.uint8)

        # Pack RGB into uint32 using vectorized bit operations
        # Format: 0x00RRGGBB
        packed_rgb = (
            (rgb_data[:, 0].astype(np.uint32) << 16)
            | (rgb_data[:, 1].astype(np.uint32) << 8)
            | rgb_data[:, 2].astype(np.uint32)
        )

        # Create structured array for efficient binary packing
        # This is much faster than manual struct.pack in a loop
        point_dtype = np.dtype(
            [
                ("x", np.float32),
                ("y", np.float32),
                ("z", np.float32),
                ("rgb", np.uint32),
            ]
        )

        # Original optical frame coordinates
        x_optical = points[:, 0].astype(np.float32)
        y_optical = points[:, 1].astype(np.float32)
        z_optical = points[:, 2].astype(np.float32)

        # Transform to a standard Z-up frame (X-forward, Y-left, Z-up)
        # New X = Optical Z
        # New Y = -Optical X
        # New Z = -Optical Y
        # Create structured array - vectorized assignment
        cloud_array = np.empty(n_points, dtype=point_dtype)
        cloud_array["x"] = z_optical
        cloud_array["y"] = -x_optical
        cloud_array["z"] = -y_optical
        # cloud_array["x"] = x_optical
        # cloud_array["y"] = y_optical
        # cloud_array["z"] = z_optical
        cloud_array["rgb"] = packed_rgb

        # Convert to bytes - much faster than individual struct packing
        cloud_data = cloud_array.tobytes()

        # Create PointCloud2 message
        pc2_msg = PointCloud2()
        pc2_msg.header = header
        pc2_msg.height = 1
        pc2_msg.width = n_points
        pc2_msg.fields = self.fields
        pc2_msg.is_bigendian = False
        pc2_msg.point_step = 16  # 4 bytes each for x, y, z, rgb
        pc2_msg.row_step = pc2_msg.point_step * n_points
        pc2_msg.data = cloud_data
        pc2_msg.is_dense = True

        return pc2_msg

    def run(self):
        """Keep the node running with optimized spin"""
        rospy.loginfo("Point cloud publisher node is running...")

        # Use a timer for periodic republishing if needed (optional)
        # Hz, 0 = no republishing
        republish_rate = rospy.get_param("~republish_rate", 0.0)

        if republish_rate > 0:
            rospy.Timer(rospy.Duration(1.0 / republish_rate), self.timer_callback)
            rospy.loginfo(f"Republishing point cloud at {republish_rate} Hz")

        rospy.spin()

    def timer_callback(self, event):
        """Timer callback for periodic republishing"""
        # Only republish if there are subscribers
        if self.pub.get_num_connections() > 0:
            self.load_and_publish_pointcloud()


if __name__ == "__main__":
    try:
        publisher = PointCloudPublisher()
        publisher.run()
    except rospy.ROSInterruptException:
        pass
