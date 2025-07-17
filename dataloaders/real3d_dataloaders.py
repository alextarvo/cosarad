""" Dataset classes for the Real3D dataset """

from torch.utils.data import Dataset
import glob
import os
import numpy as np
import logging
import constants
from . import util_dataloaders as util_dload
from . import base_dataloaders as base_dloaders
import re


class Dataset3DADTrain(base_dloaders.DatasetBaseTrain):
    """Loads a 3D point clouds from Real3D-AD training set.
    This set is fairly small, and typically contains 4 objects per class. All labels are "good"
    """

    def __init__(self, dataset_dir, class_name, normalize=True):
        """
        :param dataset_dir: the path to the Real3D-AD dataset
        :param class_name: the name of the class (plane, car, ....  - see constants.py)
        """
        super().__init__(dataset_dir, class_name, normalize)
        self.train_files_list = glob.glob(str(os.path.join(self.dataset_dir, self.class_name, 'train')) + '/*template*.pcd')
        self.train_files_list.sort()

    def __getitem__(self, idx):
        """Returns a tuple consisting of:
        - unordered point cloud as Numpy array
        - mask for each point in the pointcloud (0: good, 1: anomalous)
        - label for the whole object: 0: good; 1: anomalous
        - names of the file PC was loaded from
        """
        pcd, np_pointcloud = util_dload.read_point_cloud(self.train_files_list[idx])
        ret = self._no_anomaly_return_dict(self._scale_and_normalize(np_pointcloud), self.train_files_list[idx])
        return ret

    def __len__(self):
        return len(self.train_files_list)


def load_text_verify_against_pcd(np_pointcloud, pcd_file_name, txt_file_name):
    """Load the pointcloud with labels from the .txt file and verify the contains against the .pcd file

    Returns pointcloud in the Numpy array format. First 3 dimensions are coordinates, last dim is a label
    """
    pcd_text = np.genfromtxt(txt_file_name, delimiter=" ")
    # logging.info(f'Loaded file {txt_file_name}; dimensions: {pcd_text.shape}')
    if pcd_text.shape[1] != 4:
        logging.warning(f'Contents of {pcd_file_name} have wrong shape {pcd_text.shape}')
        return None
    pointcloud_text = pcd_text[:, :3]
    if pointcloud_text.shape != np_pointcloud.shape:
        logging.warning(f'File {pcd_file_name}; the shape {pointcloud_text.shape} of the text '
                        f'point cloud do not match the shape {np_pointcloud.shape} of the PCD point cloud')
        return None
    # if not np.allclose(np_sorted_rows(np_pointcloud), np_sorted_rows(pointcloud_text), rtol=1e-2):
    if not np.allclose(np_pointcloud[util_dload.np_sorted_idx(np_pointcloud)], pointcloud_text[util_dload.np_sorted_idx(pointcloud_text[:,:3])], rtol=1e-2):
        logging.warning(f'Contents of {pcd_file_name} do not match contents of {txt_file_name}')
        return None

    # # Sort the text-based point cloud and make sure its very close to _already sorted_ NP point cloud
    # txt_sorted_idx = np_sorted_idx(pointcloud_text)
    # pointcloud_text = pointcloud_text[txt_sorted_idx]
    # pcd_text = pcd_text[txt_sorted_idx]
    # if not np.allclose(np_pointcloud, pointcloud_text, rtol=1e-2):
    #     logging.warning(f'Contents of {pcd_file_name} do not match contents of {txt_file_name}')
    #     return None
    return pcd_text


class Dataset3DADTest(base_dloaders.DatasetBaseTest):
    def __init__(self, dataset_dir, class_name, normalize=True, full_pc=True):
        """
        :param dataset_dir: the path to the Real3D-AD dataset
        :param class_name: the name of the class (plane, car, ....  - see constants.py)

        """
        super().__init__(dataset_dir, class_name, normalize)
        all_files_names = glob.glob(str(os.path.join(self.dataset_dir, self.class_name, 'test')) + '/*.pcd')
        self.test_files_names = [os.path.splitext(os.path.basename(f))[0] for f in all_files_names]
        self.test_files_names.sort()
        self.max_radius_for_mask = 0.01
        if full_pc:
            self.test_files_names = [os.path.splitext(os.path.basename(f))[0]
                                    for f in all_files_names if (not 'cut' in f and not 'copy' in f)]
        # else:
        #     self.test_files_names = [os.path.splitext(os.path.basename(f))[0]
        #                             for f in all_files_names if 'cut' in f]
        self.regex_pattern = r'^(?:\d+)_([a-zA-Z]+)(?:_cut)?$'

    def __getitem__(self, idx):
        file_name = self.test_files_names[idx]
        input_file_name = str(os.path.join(self.dataset_dir, self.class_name, 'test', file_name)) + '.pcd'
        anomaly_mask_file_name = str(os.path.join(self.dataset_dir, self.class_name, 'gt', file_name)) + '.txt'
        # pcd = o3d.io.read_point_cloud(input_file_name)
        # np_pointcloud = np.array(pcd.points)
        pcd, np_pointcloud = util_dload.read_point_cloud(input_file_name)
        # logging.info(f'Loaded file {input_file_name}; dimensions: {np_pointcloud.shape}')
        if 'good' in file_name:
            ret = self._no_anomaly_return_dict(self._scale_and_normalize(np_pointcloud), input_file_name)
            return ret
        else:
            # Extract the defect type from the file name using regex
            match = re.match(self.regex_pattern, file_name)
            assert match, f'Cant parse file name {input_file_name} to extract the anomaly type'
            anomaly_type = match.group(1)

            # np_pc_mask = load_text_verify_against_pcd(np_pointcloud, input_file_name, anomaly_mask_file_name)
            np_pc_mask = np.genfromtxt(anomaly_mask_file_name, delimiter=" ")
            if np_pc_mask is None:
                logging.warning(f'Failed to load anomaly label for item {idx} from file {np_pc_mask}')
                return None
            if np_pc_mask.shape[1] != 4:
                logging.critical(f'Contents of {anomaly_mask_file_name} have wrong shape {np_pc_mask.shape}')
                raise ValueError(f'Contents of {anomaly_mask_file_name} have  wrong shape {np_pc_mask.shape}')
            # logging.info(f'Loaded file {anomaly_mask_file_name}; dimensions: {np_pc_mask.shape}')
            # mask = np_pc_mask[:, 3]
            np_pc_idx = np.where(np_pc_mask[:, 3] > constants.GOOD_MASK)
            anomaly_mask = util_dload.mask_defects_in_pc(np_pointcloud, np_pc_mask[np_pc_idx, 0:3][0], max_radius=self.max_radius_for_mask)
            ret = self._anomaly_return_dict(
                self._scale_and_normalize(np_pointcloud), anomaly_mask,
                input_file_name, anomaly_mask_file_name, anomaly_type)
            return ret

    def __len__(self):
        return len(self.test_files_names)
