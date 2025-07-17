"""Dataset classes MulSen anomaly detection dataset"""

from torch.utils.data import Dataset
import glob
import os
import numpy as np
import logging
import constants
from . import util_dataloaders as util_dload
from . import base_dataloaders as base_dloaders
from pathlib import Path

class DatasetMulSenTrain(base_dloaders.DatasetBaseTrain):
    """Loads a 3D point clouds from the multisensory training set.
    This set is large containing up to 90 objects
    """

    def __init__(self, dataset_dir, class_name, normalize=True, num_points=100000):
        """
        :param dataset_dir: the path to the Real3D-AD dataset
        :param class_name: the name of the class (plane, car, ....  - see constants.py)
        :param normalize: if set to true the PC will be normalized, bringing it to zero mean
        :param: num_points: if the input is the object mesh, this is the number of points to be sampled from the mesh
        """
        super().__init__(dataset_dir, class_name, normalize, num_points)
        self.train_files_list = glob.glob(str(os.path.join(self.dataset_dir, self.class_name, 'Pointcloud', 'train')) + '/*.stl')
        self.train_files_list.sort()

    def __getitem__(self, idx):
        """Returns a tuple consisting of:
        - unordered point cloud as Numpy array
        - mask for each point in the pointcloud (0: good, 1: anomalous)
        - label for the whole object: 0: good; 1: anomalous
        - name of the file PC was loaded from
        """
        # Read the mesh and convert it to the point cloud
        pcd, np_pointcloud = util_dload.read_mesh(self.train_files_list[idx], self.num_points)
        ret = self._no_anomaly_return_dict(self._scale_and_normalize(np_pointcloud), self.train_files_list[idx])
        return ret

    def __len__(self):
        return len(self.train_files_list)


class DatasetMulSenTest(base_dloaders.DatasetBaseTest):
    def __init__(self, dataset_dir, class_name, normalize=True, num_points=100000):
        super().__init__(dataset_dir, class_name, normalize, num_points)
        self.test_file_paths = glob.glob(
            str(os.path.join(self.dataset_dir, self.class_name, 'Pointcloud', 'test', '*', '*.stl')),
            recursive=True)
        self.test_file_paths.sort()

    def __getitem__(self, idx):
        """Returns a tuple consisting of:
        - unordered point cloud as Numpy array
        - mask for each point in the pointcloud (0: good, 1: anomalous)
        - label for the whole object: 0: good; 1: anomalous
        - name of the file PC was loaded from
        """
        input_file_path = self.test_file_paths[idx]
        anomaly_mask_file_path = input_file_path.replace('/test/', '/GT/').replace('.stl', '.txt')
        pcd, np_pointcloud = util_dload.read_mesh(input_file_path, self.num_points)
        # Extract the anomaly type from the name of the upper-level folder
        anomaly_type = Path(input_file_path).parent.name

        # logging.info(f'Loaded PC from file {input_file_path}; dimensions: {np_pointcloud.shape}')
        if not os.path.exists(anomaly_mask_file_path):
            if anomaly_type not in constants.mulsen_anomaly_free_types:
                logging.warning(f'GT point cloud not present for the file {anomaly_mask_file_path}; skipping the file')
                return None
            np_pointcloud, pts_removed = self._radius_outlier_removal(np_pointcloud, nb_neighbors=20, radius=2)
            ret = self._no_anomaly_return_dict(self._scale_and_normalize(np_pointcloud), input_file_path)
            ret['points_removed'] = pts_removed
            return ret
        else:
            assert anomaly_type not in constants.mulsen_anomaly_free_types,\
                f'GT point cloud is present for the anomaly-free file {input_file_path}'
            np_pc_anomalous = np.genfromtxt(anomaly_mask_file_path, delimiter=",")
            if np_pc_anomalous.shape[1] != 3:
                logging.critical(f'Contents of {anomaly_mask_file_path} have wrong shape {np_pc_anomalous.shape}')
                raise ValueError(f'Contents of {anomaly_mask_file_path} have  wrong shape {np_pc_anomalous.shape}')
            np_pointcloud, pts_removed = self._radius_outlier_removal(np_pointcloud, nb_neighbors=20, radius=2)
            anomaly_mask = util_dload.mask_defects_in_pc(np_pointcloud, np_pc_anomalous)
            ret = self._anomaly_return_dict(
                self._scale_and_normalize(np_pointcloud), anomaly_mask,
                input_file_path, anomaly_mask_file_path, anomaly_type)
            ret['points_removed'] = pts_removed
            return ret

    def __len__(self):
        return len(self.test_file_paths)
