from torch.utils.data import Dataset
import glob
import os
import numpy as np
import logging
import constants
from . import util_dataloaders as util_dload
import open3d as o3d
from collections import defaultdict

PC_NO_SCALE = 1.0


class DatasetBase(Dataset):
    def __init__(self, dataset_dir, class_name, normalize=True, num_points=100000, scale=PC_NO_SCALE):
        self.class_name = class_name
        self.dataset_dir = dataset_dir
        self.normalize = normalize
        self.num_points = num_points
        self.scale = scale

    def _scale_and_normalize(self, np_pointcloud):
        """If requested by the dataset, normalize and scale the PC"""
        if self.normalize:
            np_pointcloud = util_dload.normalize_pc(np_pointcloud)
        if self.scale != PC_NO_SCALE:
            np_pointcloud = util_dload.scale_pc(np_pointcloud, self.scale)
        return np_pointcloud

    def _statistical_outlier_removal(self, np_pointcloud, nb_neighbors, std_ratio):
        points_unfilter = np_pointcloud.shape[0]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np_pointcloud)
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        filtered_pcd = pcd.select_by_index(ind)
        filtered_np_pointcloud = np.asarray(filtered_pcd.points)
        points_filter = filtered_np_pointcloud.shape[0]
        points_removed = points_unfilter - points_filter
        assert points_removed >= 0
        return filtered_np_pointcloud, points_removed


    def _radius_outlier_removal(self, np_pointcloud, nb_neighbors, radius):
        points_unfilter = np_pointcloud.shape[0]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np_pointcloud)
        cl, ind = pcd.remove_radius_outlier(nb_points=nb_neighbors, radius=radius)
        filtered_pcd = pcd.select_by_index(ind)
        filtered_np_pointcloud = np.asarray(filtered_pcd.points)
        points_filter = filtered_np_pointcloud.shape[0]
        points_removed = points_unfilter - points_filter
        assert points_removed >= 0
        return filtered_np_pointcloud, points_removed


    def _init_no_anomaly_mask(self, np_pointcloud):
        """ return  a per-pixel mask of "correct / anomalous" for PC patches """
        return np.full((np_pointcloud.shape[0]), fill_value=constants.GOOD_MASK, dtype=np.float32)

    def _no_anomaly_return_dict(self, np_pointcloud, input_file):
        # ret_dict = defaultdict(
        #     lambda: 0,
        #     {
        #         'np_pointcloud': np_pointcloud,
        #         'point_mask': self._init_no_anomaly_mask(np_pointcloud),
        #         'object_label': constants.GOOD_MASK,
        #         'input_file': input_file,
        #         'anomaly_mask_file': '',
        #         'anomaly_type': constants.NO_ANOMALY,
        #         'scale': self.scale
        #     }
        # )
        # return ret_dict
        return {
            'np_pointcloud': np_pointcloud,
            'point_mask': self._init_no_anomaly_mask(np_pointcloud),
            'object_label': constants.GOOD_MASK,
            'input_file': input_file,
            'anomaly_mask_file': '',
            'anomaly_type': constants.NO_ANOMALY,
            'scale': self.scale,
            'points_removed': 0
        }

    def _anomaly_return_dict(self, np_pointcloud, np_anomalymask, input_file, anomaly_mask_file, anomaly_type):
        # ret_dict = defaultdict(
        #     lambda: 0,
        #     {
        #     'np_pointcloud': np_pointcloud,
        #     'point_mask': np_anomalymask,
        #     'object_label': constants.ANOMALY_MASK,
        #     'input_file': input_file,
        #     'anomaly_mask_file': anomaly_mask_file,
        #     'anomaly_type': anomaly_type,
        #     'scale': self.scale
        #     }
        # )
        # return ret_dict
        return {
            'np_pointcloud': np_pointcloud,
            'point_mask': np_anomalymask,
            'object_label': constants.ANOMALY_MASK,
            'input_file': input_file,
            'anomaly_mask_file': anomaly_mask_file,
            'anomaly_type': anomaly_type,
            'scale': self.scale,
            'points_removed': 0
        }


class DatasetBaseTrain(DatasetBase):
    def __init__(self, dataset_dir, class_name, normalize=True, num_points=100000, scale=PC_NO_SCALE):
        super().__init__(dataset_dir, class_name, normalize, num_points, scale)


class DatasetBaseTest(DatasetBase):
    def __init__(self, dataset_dir, class_name, normalize=True, num_points=100000, scale=PC_NO_SCALE):
        super().__init__(dataset_dir, class_name, normalize, num_points, scale)
