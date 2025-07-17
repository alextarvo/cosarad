import copy

import dataloaders.dl_dataloaders
import dataloaders.util_dataloaders

import math

import open3d as o3d
from torch.utils.data import DataLoader
# from feature_extractors.fpfh_pc_features import PC_FPFHFeatures
import numpy as np

import argparse
import constants
from dataloaders import real3d_dataloaders as dloaders

voxel_size = 0.5

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcd_path", type=str, default=None,
                        help='Path to the Real3DAD dataset to visualize')
    parser.add_argument("--pcd_file", type=str, default=None,
                        help='Path to the single .pcd file to visualize')
    parser.add_argument("--registered_pcd_path", type=str, default=None,
                        help='Path to the single .pcd file to visualize')
    return parser.parse_args()


def show_pointcloud_grid(batch, spacing=50.0):
    """
    Display multiple point clouds in a 2D grid layout using Open3D.

    Args:
        batch: dict containing 'np_pointcloud', a list of [N, 3] arrays wrapped in a list.
        spacing: float, how far apart point clouds are placed in the grid.
    """
    pointclouds = batch['np_pointcloud']
    total = len(pointclouds)

    # Compute number of rows and columns (closest to square)
    cols = math.ceil(math.sqrt(total))
    rows = math.ceil(total / cols)

    pcds = []

    for idx, pc in enumerate(pointclouds):
        if pc.shape[1] != 3:
            raise ValueError(f"Point cloud {idx} is not [N, 3], got {pc.shape}")

        row = idx // cols
        col = idx % cols
        offset = np.array([col * spacing, -row * spacing, 0])

        pc_translated = pc + offset
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_translated)
        # pcd.paint_uniform_color(np.random.rand(3))  # optional random color
        pcds.append(pcd)
    o3d.visualization.draw_geometries(pcds)

def show_registered_pointcloud_grid(template, pc_batch, spacing=50.0):
    """
    Display multiple [registered] point clouds in a 2D grid layout using Open3D.

    Args:
        template: the PC template for a registered point cloud
        pc_batch: batches of pointcloud.
        spacing: float, how far apart point clouds are placed in the grid.
    """
    total = len(pc_batch)

    # Compute number of rows and columns (closest to square)
    cols = math.ceil(math.sqrt(total))
    rows = math.ceil(total / cols)

    pcds = []

    for idx, pc in enumerate(pc_batch):
        # if pc.shape[1] != 3:
        #     raise ValueError(f"Point cloud {idx} is not [N, 3], got {pc.shape}")

        row = idx // cols
        col = idx % cols
        offset = np.array([col * spacing, -row * spacing, 0])

        template_copy = copy.deepcopy(template)
        pc.translate(offset)
        template_copy.translate(offset)
        pc.paint_uniform_color([1, 0.706, 0])
        template_copy.paint_uniform_color([0, 0.651, 0.929])
        pcds.append(pc)
        pcds.append(template_copy)
    o3d.visualization.draw_geometries(pcds)

def show_unregistered_and_registered_triplet(
        template_pc, good_pc_unreg, bad_pc_unreg,
        good_pc_reg, bad_pc_reg, spacing=50.0, window_title=''):
    """Given a triplet of PCs - template, registered, unregistered -
    visualize them in a single grid"""

    def translate_paint_append(pcs, pc, translation, color=None):
        # For a point cloud, translate it with a translation,
        # paint it with a color, and add to the list pcs of the point clouds
        pc_copy = copy.deepcopy(pc)
        pc_copy.translate(translation)
        if color is not None:
            pc_copy.paint_uniform_color(color)
        pcs.append(pc_copy)

    pcds = []
    # First row: original (un-registered) PCs
    translate_paint_append(pcds, template_pc, np.array([0, 0, 0]), [0, 0.651, 0.929])
    translate_paint_append(pcds, good_pc_unreg, np.array([spacing, 0, 0]), [0, 1, 0])
    translate_paint_append(pcds, bad_pc_unreg, np.array([2*spacing, 0, 0]), [1, 0, 0])

    # First row: registered PCs
    translate_paint_append(pcds, template_pc, np.array([0, -spacing, 0]), [0, 0.651, 0.929])
    translate_paint_append(pcds, good_pc_reg, np.array([spacing, -spacing, 0]), [0, 1, 0])
    translate_paint_append(pcds, bad_pc_reg, np.array([2*spacing, -spacing, 0]), [1, 0, 0])

    # Third row: un-registered+registered PCs in the same frame
    translate_paint_append(pcds, template_pc, np.array([spacing, -2*spacing, 0]), [0, 0.651, 0.929])
    translate_paint_append(pcds, good_pc_reg, np.array([spacing, -2*spacing, 0]), [0, 1, 0])
    translate_paint_append(pcds, template_pc, np.array([2*spacing, -2*spacing, 0]), [0, 0.651, 0.929])
    translate_paint_append(pcds, bad_pc_reg, np.array([2*spacing, -2*spacing, 0]), [1, 0, 0])
    o3d.visualization.draw_geometries(pcds, window_name=window_title)


def show_patch_triplet(
        template_pc, good_pc, bad_pc,
        anchor_patch, good_patch, bad_patch,
        spacing=50.0, window_title=''):
    """Given a triplet of PCs - template, registered, unregistered -
    visualize them in a single grid"""

    def translate_paint_append(pcs, pc, translation, color=None):
        # For a point cloud, translate it with a translation,
        # paint it with a color, and add to the list pcs of the point clouds
        pc_copy = copy.deepcopy(pc)
        pc_copy.translate(translation)
        if color is not None:
            pc_copy.paint_uniform_color(color)
        pcs.append(pc_copy)

    pcds = []
    # First row: original (un-registered) PCs
    translate_paint_append(pcds, template_pc, np.array([0, 0, 0]), [0, 0, 0.5])
    translate_paint_append(pcds, good_pc, np.array([spacing, 0, 0]), [0, 0.5, 0])
    translate_paint_append(pcds, bad_pc, np.array([2*spacing, 0, 0]), [0.5, 0, 0])
    translate_paint_append(pcds, anchor_patch, np.array([0, 0, 0]), [0, 0, 1])
    translate_paint_append(pcds, good_patch, np.array([spacing, 0, 0]), [0, 1, 0])
    translate_paint_append(pcds, bad_patch, np.array([2*spacing, 0, 0]), [1, 0, 0])


    # First row: registered PCs
    translate_paint_append(pcds, anchor_patch, np.array([0, -spacing, 0]), [0, 0, 1])
    translate_paint_append(pcds, good_patch, np.array([spacing, -spacing, 0]), [0, 1, 0])
    translate_paint_append(pcds, bad_patch, np.array([2*spacing, -spacing, 0]), [1, 0, 0])

    # Third row: un-registered+registered PCs in the same frame
    translate_paint_append(pcds, anchor_patch, np.array([spacing, -2*spacing, 0]), [0, 0, 1])
    translate_paint_append(pcds, good_patch, np.array([spacing, -2*spacing, 0]), [0, 1, 0])
    translate_paint_append(pcds, anchor_patch, np.array([2*spacing, -2*spacing, 0]), [0, 0, 1])
    translate_paint_append(pcds, bad_patch, np.array([2*spacing, -2*spacing, 0]), [1, 0, 0])
    o3d.visualization.draw_geometries(pcds, window_name=window_title)


if __name__ == "__main__":
    args = get_args()
    if args.pcd_file is not None:
        pcd = o3d.io.read_point_cloud(args.pcd_path)
        vis.summary_pointcloud(pcd)
        o3d.visualization.draw_geometries([pcd])
    elif args.pcd_path is not None:
        for real_class in constants.real3d_object_classes:
            print(f'Testing object class {real_class}')
            train_loader = DataLoader(
                dloaders.Dataset3DADTrain(args.pcd_path, real_class, normalize=True),
                batch_size=4, shuffle=False, drop_last=True, collate_fn=dataloaders.util_dataloaders.dloader_dict_collate)
            for idx, batch in enumerate(train_loader):
                show_pointcloud_grid(batch)

            train_loader = DataLoader(
                dloaders.Dataset3DADTest(args.pcd_path, real_class, normalize=True, full_pc=False),
                batch_size=16, shuffle=False, drop_last=True, collate_fn=dataloaders.util_dataloaders.dloader_dict_collate)
            for idx, batch in enumerate(train_loader):
                show_pointcloud_grid(batch)
    elif args.registered_pcd_path is not None:
        ds_registration_vis = dataloaders.dl_dataloaders.DatasetRegistrationVisualize(args.registered_pcd_path)
        registered_pc_loader = DataLoader(
            ds_registration_vis,
            batch_size=9, shuffle=False, drop_last=True, collate_fn=dataloaders.dl_dataloaders.collate_fn_pc_list)
        for idx, batch in enumerate(registered_pc_loader):
            show_registered_pointcloud_grid(ds_registration_vis.get_template(), batch)


    else:
        print('Either pcd_file or pcd_path must be specified')