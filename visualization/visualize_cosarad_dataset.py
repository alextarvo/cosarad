"""File that visulizes the whole cosarad dataset"""

import argparse
import numpy as np
from torch.utils.data import DataLoader

import dataloaders.cosarad_dataloaders as cosarad_dloaders
import dataloaders.util_dataloaders as util_dloaders

import open3d as o3d
import os

import constants

def load_input_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    pc = None
    if ext == '.pcd':
        pc = o3d.io.read_point_cloud(file_path)
    elif ext in ['.obj', '.stl']:
        pc = o3d.io.read_triangle_mesh(file_path)
        pc.compute_vertex_normals()
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    centroid = pc.get_center()  # returns a 3D vector
    pc.translate(-centroid)
    return pc

def create_colored_pointcloud(np_points, mask=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_points)

    if mask is None:
        # Light blue color
        colors = np.tile([0.6, 0.8, 1.0], (np_points.shape[0], 1))
    else:
        # Light blue for normal, light red for anomalies
        colors = np.tile([0.6, 0.8, 1.0], (np_points.shape[0], 1))
        colors[mask == 1] = [1.0, 0.6, 0.6]

    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def translate_geometry(geometry, x=0, y=0, z=0):
    return geometry.translate((x, y, z), relative=False)

def visualize_data(template_data, train_data):
    spacing = 1.2  # for layout

    # Load input files
    template_input = load_input_file(template_data['input_file'][0])
    train_input = load_input_file(train_data['input_file'][0])

    # Convert np_pointclouds
    template_pcd = create_colored_pointcloud(template_data['np_pointcloud'][0])
    train_pcd = create_colored_pointcloud(train_data['np_pointcloud'][0], train_data['point_mask'][0])

    # Normalize everything for layout
    all_geoms = [template_input, template_pcd, train_input, train_pcd]
    dims = np.array([p.get_axis_aligned_bounding_box().get_extent() for p in all_geoms])
    max_dims = dims.max(axis=0)
    dx = max_dims[0] * spacing
    dy = max_dims[1] * spacing

    # Layout positions
    template_input.translate((0, 0, 0))
    template_pcd.translate((dx, 0, 0))
    train_input.translate((0, dy, 0))
    train_pcd.translate((dx, dy, 0))

    # Add file labels as geometry (simple text as 3D geometry is not trivial in Open3D, so we skip it here)
    print(f"Template npz file: {template_data['npz_file'][0]}")
    print(f"Template input file: {template_data['input_file'][0]}")
    print(f"Train npz file: {train_data['npz_file'][0]}")
    print(f"Train input file:    {train_data['input_file'][0]}")

    # Launch visualizer
    o3d.visualization.draw_geometries(
        [template_input, template_pcd, train_input, train_pcd],
        window_name=f"npz train: {train_data['npz_file'][0]}; Template: {template_data['input_file'][0]}; Train: {train_data['input_file'][0]}",
        width=2000,
        height=1400
    )

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cosarad_data_path", type=str, default=None,
                        help='Path to the complete COSARAD dataset to visualize')
    parser.add_argument("--dataset_filter", type=str, default='*',
                        help='Name template (a filter) for names of the datasets to be visualized: '
                             'real3dad, ashapenet, mulsen')
    parser.add_argument("--class_filter", type=str, default='*',
                        help='Name template (a filter) for names of the object classes to be visualized')
    parser.add_argument("--index_filter", type=str, default='*',
                        help='Name template for the indices of the objects to be visualized')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    template_loader = DataLoader(
        cosarad_dloaders.DatasetCosarad(args.cosarad_data_path, 'template', args.dataset_filter, args.class_filter, '.*', '.*'),
        num_workers=0, batch_size=1, shuffle=False, drop_last=False, collate_fn=util_dloaders.dloader_dict_collate)
    train_loader = DataLoader(
        cosarad_dloaders.DatasetCosarad(args.cosarad_data_path, 'train', args.dataset_filter, args.class_filter, '.*', args.index_filter),
        num_workers=0, batch_size=1, shuffle=False, drop_last=False, collate_fn=util_dloaders.dloader_dict_collate)

    template_iter = iter(template_loader)
    train_iter = iter(train_loader)

    template_item = next(template_iter)
    train_item = next(train_iter)

    end_template = False
    end_train = False

    while not (end_template and end_train):
        visualize_data(template_item, train_item)

        try:
            template_item = next(template_iter)
        except StopIteration:
            end_template = True

        try:
            train_item = next(train_iter)
        except StopIteration:
            end_train = True
