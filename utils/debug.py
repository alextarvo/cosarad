import os
import open3d as o3d
import sys
import numpy as np

def in_debugger():
    return sys.gettrace() is not None


def np_point_cloud_stats(pcd_np, name):
    """
    Prints diagnostic info about a point cloud:
    - Number of points
    - Axis-aligned bounding box size (max dimensions)
    - Diagonal length of bounding box
    """
    if pcd_np.shape[0] == 0:
        print(f"[{name}] Empty point cloud.")
        return

    min_bounds = pcd_np.min(axis=0)
    max_bounds = pcd_np.max(axis=0)
    dims = max_bounds - min_bounds
    diagonal = np.linalg.norm(dims)

    print(f"Point cloud [{name}] Info:")
    print(f"  Num. points     : {pcd_np.shape[0]}")
    print(f"  X range         : {min_bounds[0]:.4f} to {max_bounds[0]:.4f}  (delta = {dims[0]:.4f})")
    print(f"  Y range         : {min_bounds[1]:.4f} to {max_bounds[1]:.4f}  (delta = {dims[1]:.4f})")
    print(f"  Z range         : {min_bounds[2]:.4f} to {max_bounds[2]:.4f}  (delta = {dims[2]:.4f})")
    print(f"  Bounding box diagonal: {diagonal:.4f}")



def save_registered_pointclouds(base_path, subfolder, split, idx, basic_template, registered_np):
    """ Save a pair of registertered point clouds to the disk
    base_path: path to the output folder
    subfolder: the name of the object
    split: train vs. test

    """
    # Construct full directory path
    output_dir = os.path.join(base_path, subfolder)
    os.makedirs(output_dir, exist_ok=True)  # Create directories if they don't exist

    # Define file names using idx
    template_path = os.path.join(output_dir, f"template.pcd")
    registered_path = os.path.join(output_dir, f"registered_{split}_{idx}.pcd")

    # Save point clouds
    # if not os.path.exists(template_path):
    if not os.path.exists(template_path):
        basic_template_pc = o3d.geometry.PointCloud()
        basic_template_pc.points = o3d.utility.Vector3dVector(basic_template)
        o3d.io.write_point_cloud(template_path, basic_template_pc)

    registered_pc = o3d.geometry.PointCloud()
    registered_pc.points = o3d.utility.Vector3dVector(registered_np)
    o3d.io.write_point_cloud(registered_path, registered_pc)
