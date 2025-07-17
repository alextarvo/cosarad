"""Dataset classes for internal datasets used for COSARAD"""

import glob
import logging
import os

import numpy as np
import open3d as o3d
import pandas as pd

from torch.utils.data import Dataset

from dataloaders import util_dataloaders as util_dload


def collate_fn_pc_list(batch):
    """
    Collate function for batching point clouds.
    Each item in the batch is an open3d.geometry.PointCloud.
    """
    return batch  # just returns the list of PointClouds


class DatasetRegistrationVisualize(Dataset):
    """ A dataloader specifically to load and visualize the registered point clouds"""

    def __init__(self, dataset_dir):
        """
        :param dataset_dir: the path to the Real3D-AD dataset
        """
        # self.class_name = class_name
        self.dataset_dir = dataset_dir
        template_files_name = glob.glob(os.path.join(dataset_dir, 'template.pcd'))
        assert (len(template_files_name) == 1)
        # self.pcd_template = o3d.io.read_point_cloud(template_files_name[0])
        self.pcd_template, _ = util_dload.read_point_cloud(template_files_name[0])
        self.registered_files_names = glob.glob(dataset_dir + '/registered*.pcd')

    def get_template(self):
        return self.pcd_template

    def __getitem__(self, idx):
        # pcd_registered = o3d.io.read_point_cloud(self.registered_files_names[idx])
        pcd_registered, _ = util_dload.read_point_cloud(self.registered_files_names[idx])
        assert (pcd_registered is not None)
        return pcd_registered

    def __len__(self):
        return len(self.registered_files_names)


def triplet_dict_collate(batch_in):
    if batch_in is None or batch_in[0] is None:
        return {
            'np_pointcloud_anchor': None,
            'np_pointcloud_good': None,
            'np_pointcloud_bad': None,
            'pcd_anchor': None,
            'pcd_good': None,
            'pcd_bad': None,
            'mask_anchor': None,
            'mask_good': None,
            'mask_bad': None,
            'fname_anchor': None,
            'fname_good': None,
            'fname_bad': None,
        }
    return {
        key: [d[key] for d in batch_in if d is not None] for key in batch_in[0]
    }


class TripletPCDataset(Dataset):
    """A dataset for a triplet patch generator"""

    def __init__(self, transform_csv_path, dataset_dir, class_name, normalize=True, threshold_mean_dist=1.0,
                 threshold_max_dist=10.0):
        """transform_csv_path: a path to the .csv file with names of .pcd files for two good and one bad PCs.
            also contains transformation matrices for their registration
        """
        self.df_reg_pcd_triplets = pd.read_csv(transform_csv_path, header=0, index_col=0)
        self.dataset_dir = dataset_dir
        self.class_name = class_name
        self.normalize = normalize

        self.df_reg_pcd_triplets['good_good_transform'] = (
            self.df_reg_pcd_triplets['good_good_transform'].apply(util_dload.string_to_matrix))
        self.df_reg_pcd_triplets['good_bad_transform'] = (
            self.df_reg_pcd_triplets['good_bad_transform'].apply(util_dload.string_to_matrix))

        # Verify the registration quality and drop the worst ones
        self.df_reg_pcd_triplets = self.__drop_rows_over_threshold(
            self.df_reg_pcd_triplets, 'good_good_mean_dist', threshold_mean_dist)
        self.df_reg_pcd_triplets = self.__drop_rows_over_threshold(
            self.df_reg_pcd_triplets, 'good_bad_mean_dist', threshold_mean_dist)
        # self.__drop_rows_over_threshold(self.df_reg_pcd_triplets, 'good_good_max_dist', threshold_max_dist)
        # self.__drop_rows_over_threshold(self.df_reg_pcd_triplets, 'good_bad_max_dist', threshold_max_dist)

    def __drop_rows_over_threshold(self, df, column_name, threshold):
        """Drops rows from the dataset whose values in the column exceed a given threshold"""
        initial_len = len(df)
        df = df[df[column_name] <= threshold]  # keep only rows <= threshold
        num_dropped_rows = initial_len - len(df)
        if num_dropped_rows > 0:
            logger.warning(f'Dropped {num_dropped_rows} rows whose {column_name} values exceed {threshold}')
        return df

    def __getitem__(self, idx):
        # Load Point clouds in the .txt format
        fname_anchor = os.path.join(
            self.dataset_dir, self.class_name, 'test', self.df_reg_pcd_triplets.iloc[idx]['template_file'])
        fname_good = os.path.join(
            self.dataset_dir, self.class_name, 'test', self.df_reg_pcd_triplets.iloc[idx]['good_file'])
        # Note: here we decided not to load anomalous PC- since we have to load .txt file anyways
        # fname_bad = os.path.join(
        #     self.dataset_dir, self.class_name, 'test', self.df_reg_pcd_triplets.iloc[idx]['bad_file'])
        pcd_anchor, np_pointcloud_anchor = util_dload.read_point_cloud(fname_anchor)
        pcd_good, np_pointcloud_good = util_dload.read_point_cloud(fname_good)

        # Get pointcloud labels for the anomalouys PC. For good ones all the points as 0's
        # Note: we must load the anomalous PC from the .txt file. Its points not always correspond
        # to those in .pcd files. So we use .txt file with labels directly
        fname_bad_txt = os.path.join(
            self.dataset_dir, self.class_name, 'gt',
            self.df_reg_pcd_triplets.iloc[idx]['bad_file'].replace('pcd', 'txt'))
        np_pointcloud_bad = np.genfromtxt(fname_bad_txt, delimiter=" ")
        # We never really re-arrange points, so here we extract labels here explicitly
        np_bad_labels = np_pointcloud_bad[:,3]
        np_pointcloud_bad = np_pointcloud_bad[:,:3]
        pcd_bad = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np_pointcloud_bad))
        if np_pointcloud_bad is None or np_pointcloud_bad.shape[0] == 0:
            logging.warning(f'Failed to load text label for bad, item {idx}')
            return None

        # Follow the order of the operations in our main pipeline.
        # First normalize the PCs if requested (usually we do so)
        if self.normalize:
            np_pointcloud_anchor = util_dload.normalize_pc(np_pointcloud_anchor)
            pcd_anchor.points = o3d.utility.Vector3dVector(np_pointcloud_anchor)
            np_pointcloud_good = util_dload.normalize_pc(np_pointcloud_good)
            pcd_good.points = o3d.utility.Vector3dVector(np_pointcloud_good)
            np_pointcloud_bad[:,:3] = util_dload.normalize_pc(np_pointcloud_bad)
            pcd_bad.points = o3d.utility.Vector3dVector(np_pointcloud_bad)

        # Now apply the precomputed registration transform
        # note: we assume the ordering of the points will be the same!
        pcd_good.transform(self.df_reg_pcd_triplets.iloc[idx]['good_good_transform'])
        pcd_bad.transform(self.df_reg_pcd_triplets.iloc[idx]['good_bad_transform'])
        return {
            'np_pointcloud_anchor': np.array(pcd_anchor.points),
            'np_pointcloud_good': np.array(pcd_good.points),
            'np_pointcloud_bad': np.array(pcd_bad.points),
            # Stupid Open3D can't serialize clouds, and I can't return them from a reader!
            'np_bad_labels': np_bad_labels,
            'fname_anchor': fname_anchor,
            'fname_good': fname_good,
            'fname_bad': fname_bad_txt,
        }

    def __len__(self):
        return len(self.df_reg_pcd_triplets)
