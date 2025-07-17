"""Visualizes the PC triplets registration: template (good), good, anomalous"""
import os
import pandas as pd
import open3d as o3d
import numpy as np
import copy
import argparse

import visualize_pc as vpc
import dataloaders.cosarad_dataloaders as cosarad_dloaders
import dataloaders.util_dataloaders as util_dloaders


def get_args():
    """
       Sets up command line arguments parsing
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--cosarad_data_path", type=str,
                        help='Path where the registration results, and the processed PCs in .npz files, are stored')
    parser.add_argument("--object_class", type=str, default=None,
                        help='Name of the class to run the program upon')
    return parser.parse_args()


def visualize_registration(df, row_index, cosarad_data_path):
    if row_index >= len(df):
        print(f"Row index {row_index} out of range")
        return

    # Get data from the row
    row = df.iloc[row_index]

    # Read .npz files that contain point clouds. NOTE: these should be already normalized
    train_dir = os.path.join(cosarad_data_path, 'train')
    template_npz = cosarad_dloaders.load_npz_as_native_dict(os.path.join(train_dir, row['template_file']))
    good_npz = cosarad_dloaders.load_npz_as_native_dict(os.path.join(train_dir, row['good_file']))
    bad_npz = cosarad_dloaders.load_npz_as_native_dict(os.path.join(train_dir, row['bad_file']))

    # Do normalization (enabled by default)
    # np_pointcloud_template = normalize_pc(np.array(o3d_pc_template.points))
    o3d_pc_template = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(template_npz['np_pointcloud']))
    # np_pointcloud_good = normalize_pc(np.array(o3d_pc_good.points))
    o3d_pc_good = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(good_npz['np_pointcloud']))
    # np_pointcloud_bad = normalize_pc(np_bad_points)
    o3d_pc_bad = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(bad_npz['np_pointcloud']))

    # 4. Read transformation matrices
    good_transform = util_dloaders.string_to_matrix(row['good_good_transform'])
    bad_transform = util_dloaders.string_to_matrix(row['good_bad_transform'])

    # 5. Visualize good-good registration
    good_pc_registered = copy.deepcopy(o3d_pc_good)
    good_pc_registered.transform(good_transform)
    # template_pc_copy.paint_uniform_color([1, 0, 0])  # Red
    # good_pc_registered.paint_uniform_color([0, 1, 0])      # Green
    # o3d.visualization.draw_geometries([template_pc_copy, good_pc_copy],
    #                                 window_name="Good-Good Registration",
    #                                 width=800,
    #                                 height=600)

    # 6. Visualize good-bad registration
    bad_pc_registered = copy.deepcopy(o3d_pc_bad)
    bad_pc_registered.transform(bad_transform)

    print("\nRegistration stats:")
    print(f'Template: {row["template_file"]}, Good: {row["good_file"]}, Bad: {row["bad_file"]}')
    print(f'Template-to-Good distances. Mean: {row["good_good_mean_dist"]:.4f}, '
          f'Max: {row["good_good_max_dist"]:.4f}, Chamfer: {row["good_good_chamfer"]:.4f}')
    print(f'Template-to-Bad distances. Mean: {row["good_bad_mean_dist"]:.4f}, '
          f'Max: {row["good_bad_max_dist"]:.4f}, Chamfer: {row["good_bad_chamfer"]:.4f}')

    # Visualize a triplet - first unregistered, then registered (separately) and then registered overlap
    vpc.show_unregistered_and_registered_triplet(
        o3d_pc_template, o3d_pc_good, o3d_pc_bad, good_pc_registered, bad_pc_registered)


if __name__ == "__main__":
    args = get_args()
    registration_file = f"registration_data_{args.object_class}.csv"
    registration_full_path = os.path.join(args.cosarad_data_path, 'registered_pc_triplets', registration_file)
    row_index = 0  # idx 0 is the first dataSet, and then continue....

    # 1. Read CSV file
    df = pd.read_csv(registration_full_path)
    for row_index in range(df.shape[0]):
        visualize_registration(df, row_index, args.cosarad_data_path)

