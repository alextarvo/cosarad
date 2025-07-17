"""Dataset classes Anomaly-ShapeNet anomaly detection dataset"""

from torch.utils.data import Dataset
import glob
import os
import numpy as np
import logging
import pandas as pd
import constants
from . import util_dataloaders as util_dload
from . import base_dataloaders as base_dloaders
from pathlib import Path
import re

ASHAPENET_DEFAULT_SCALE=20.0

class DatasetAnomalyShapenetTrain(base_dloaders.DatasetBaseTrain):
    """Loads a 3D point clouds from Real3D-AD training set.
    This set is fairly small, and typically contains 4 objects per class. All labels are "good"
    """

    def __init__(self, dataset_dir, class_name, normalize=True, num_points=100000, scale=ASHAPENET_DEFAULT_SCALE):
        """
        :param dataset_dir: the path to the Real3D-AD dataset
        :param class_name: the name of the class (plane, car, ....  - see constants.py)
        """
        super().__init__(dataset_dir, class_name, normalize, num_points, scale)
        self.train_files_list = glob.glob(str(os.path.join(self.dataset_dir, 'obj', self.class_name)) + '/*template*.obj')
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


class DatasetAnomalyShapenetTest(base_dloaders.DatasetBaseTest):
    def __init__(self, dataset_dir, class_name, normalize=True, num_points=100000, scale=ASHAPENET_DEFAULT_SCALE):
        super().__init__(dataset_dir, class_name, normalize, num_points, scale)
        all_files_names = glob.glob(str(os.path.join(self.dataset_dir, 'obj', self.class_name)) + '/*.obj')
        self.test_files_names = [os.path.splitext(os.path.basename(f))[0] for f in all_files_names if not 'template' in f]
        self.test_files_names.sort()
        self.regex_pattern = r'^(?:[a-zA-Z]+\d+)_([a-zA-Z]+)(?:\d+)$'

    def __getitem__(self, idx):
        input_file_name = self.test_files_names[idx]
        # Extract the anomaly type from the file name using regex
        obj_file_name = str(os.path.join(self.dataset_dir,  'obj', self.class_name, input_file_name)) + '.obj'
        match = re.match(self.regex_pattern, input_file_name)
        assert match, f'Cant parse file name {input_file_name} to extract the anomaly type'
        anomaly_type = match.group(1)

        anomaly_mask_file_name = str(os.path.join(self.dataset_dir, 'pcd', self.class_name, 'GT', input_file_name)) + '.txt'
        pcd, np_pointcloud = util_dload.read_mesh(obj_file_name, self.num_points)
        # logging.info(f'Loaded PC from file {obj_file_name}; dimensions: {np_pointcloud.shape}')
        if 'positive' in obj_file_name:
            # Anomaly-free PC
            ret = self._no_anomaly_return_dict(self._scale_and_normalize(np_pointcloud), obj_file_name)
            return ret
        else:
            np_pc_mask = np.genfromtxt(anomaly_mask_file_name, delimiter=",")
            if np_pc_mask.shape[1] != 4:
                logging.critical(f'Contents of {anomaly_mask_file_name} have wrong shape {np_pc_mask.shape}')
                raise ValueError(f'Contents of {anomaly_mask_file_name} have  wrong shape {np_pc_mask.shape}')
            # Fetch only anomalous points
            np_pc_idx = np.where(np_pc_mask[:, 3] > constants.GOOD_MASK)
            anomaly_mask = util_dload.mask_defects_in_pc(np_pointcloud, np_pc_mask[np_pc_idx, 0:3][0])
            ret = self._anomaly_return_dict(
                self._scale_and_normalize(np_pointcloud),
                anomaly_mask,
                obj_file_name, anomaly_mask_file_name, anomaly_type)
            return ret

    def __len__(self):
        return len(self.test_files_names)
