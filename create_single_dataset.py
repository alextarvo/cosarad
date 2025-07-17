"""Merges multiple anomaly detection datasets (Real3D-AD, Anomaly-ShapeNet, MulSen) into a single dataset
for future processing
"""
import os
import copy
import logging
import argparse

from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

from torch.utils.data import DataLoader

import dataloaders.util_dataloaders as util_dloaders
import dataloaders.mulsen_dataloaders as mulsen_dloaders
import dataloaders.shapenet_dataloaders as shapenet_dloaders
import dataloaders.real3d_dataloaders as real3d_dloaders
from utils.debug import in_debugger
import constants

num_workers = 0 if in_debugger() else 4

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)
logging.basicConfig(level=logging.INFO)

TEMPLATE_SPLIT = 'template'
TRAIN_SPLIT = 'train'

# The maximum percentage of anomalous points in the PC.
# PCs exceeding this threshold are considered mislabeled and are discarded
ANOMALY_PERCENT_THRESHOLD = 50

REAL3D_DS_NAME = 'real3dad'
MULSEN_DS_NAME = 'mulsen'
SHAPENET_DS_NAME = 'ashapenet'

TARGET_GOOD_TO_ANOMALY_PROPORTION = 2.0

SUMMMARY_COLUMNS = [
    'dataset',
    'object',
    'index',
    'num_points',
    'scale',
    'anomaly_type',
    'pc_bbox_min',
    'pc_bbox_max',
    'pc_bbox_dims',
    'pc_diagonal',
    'anomaly_bbox_min',
    'anomaly_bbox_max',
    'anomaly_diagonal',
    'anomaly_percent',
    'points_removed',
    'output_file',
    'input_file',
    'anomaly_mask_file'
]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

def validate_item(dataset_name, item):
    """Ensures that the item received from the dataset has all the fields set correctly."""
    if item['np_pointcloud'] is None:
        return False
    # assert item['np_pointcloud'] is not None, f'Failed to load batch from the {dataset_name} dataset!'
    assert len(item['np_pointcloud']) == 1, f'We expect a single entry read from the {dataset_name} dataset!'
    assert item['np_pointcloud'][0] is not None, f'Failed to load point cloud from the {dataset_name} dataset!'
    assert item['point_mask'][0] is not None, f'Failed to load point mask from the {dataset_name} dataset!'
    assert item['object_label'][0] is not None, f'Failed to load object label from the {dataset_name} dataset!'
    assert item['input_file'][0] is not None, f'Failed to load input file name from the {dataset_name} dataset!'
    assert item['anomaly_mask_file'][0] is not None, f'Failed to load anomaly mask file name from the {dataset_name} dataset!'
    assert item['anomaly_type'][0] is not None, f'Failed to load anomaly type from the {dataset_name} dataset!'
    return True

def save_output_file(out_dir, dataset_name, object_name, index, item, pd_summary):
    """ Outputs the PC file into the single dataset. Also collects summary in Pandas dataframe. """
    np_pointcloud = item['np_pointcloud'][0]
    point_mask = item['point_mask'][0]
    anomaly_type = item['anomaly_type'][0]
    assert (item['np_pointcloud'][0].shape[0] == item['point_mask'][0].shape[0])
    out_fpath = os.path.join(out_dir, f'{dataset_name}_{object_name}_{index}_{anomaly_type}.npz')
    os.makedirs(out_dir, exist_ok=True)

    # Full bounding box for a PC
    bbox_min = np_pointcloud.min(axis=0)
    bbox_max = np_pointcloud.max(axis=0)
    bbox_dims = bbox_max - bbox_min
    # Anomalous part bounding box
    anomalous_points = np_pointcloud[point_mask==1]
    if len(anomalous_points) > 0:
        abbox_min = anomalous_points.min(axis=0)
        abbox_max = anomalous_points.max(axis=0)
        abbox_dims = abbox_max - abbox_min
        anomaly_diagonal = np.linalg.norm(abbox_dims)
    else:
        abbox_min = np.array([0.0, 0.0, 0.0])
        abbox_max = np.array([0.0, 0.0, 0.0])
        abbox_dims = np.array([0.0, 0.0, 0.0])
        anomaly_diagonal = 0
    # Percentage of anomalous points
    anomaly_ratio = np.sum(point_mask) / len(point_mask) * 100
    if anomaly_ratio > ANOMALY_PERCENT_THRESHOLD:
        logging.warning(f'The percentage of anomalous points in a dataset {anomaly_ratio} exceeds the threshold {ANOMALY_PERCENT_THRESHOLD}')
        return pd_summary
    np.savez(out_fpath,
             np_pointcloud=np_pointcloud, point_mask=item['point_mask'][0],
             object_label=item['object_label'][0], input_file=item['input_file'][0],
             anomaly_mask_file=item['anomaly_mask_file'][0], anomaly_type=item['anomaly_type'][0],
             )
    row = {
        'dataset': dataset_name,
        'object': object_name,
        'index': index,
        'num_points': np_pointcloud.shape[0],
        'anomaly_type': anomaly_type,
        'pc_bbox_min': bbox_min.tolist(),
        'pc_bbox_max': bbox_max.tolist(),
        'pc_bbox_dims': bbox_dims.tolist(),
        'pc_diagonal': np.linalg.norm(bbox_dims),
        'anomaly_bbox_min': abbox_min.tolist(),
        'anomaly_bbox_max': abbox_max.tolist(),
        'anomaly_diagonal': anomaly_diagonal,
        'anomaly_percent': anomaly_ratio,
        'points_removed': item['points_removed'][0],
        'output_file': out_fpath,
        'input_file': item['input_file'][0],
        'anomaly_mask_file': item['anomaly_mask_file'][0]
    }
    pd_summary = pd.concat([pd_summary, pd.DataFrame([row])], ignore_index=True)
    return pd_summary


def pipeline(args):
    """The main pipeline for processing registration."""
    os.makedirs(args.output_data_path, exist_ok=True)
    out_template_dir = os.path.join(args.output_data_path, TEMPLATE_SPLIT)
    out_train_dir = os.path.join(args.output_data_path, TRAIN_SPLIT)

    os.makedirs(args.summary_output_dir, exist_ok=True)
    summary_file_train = os.path.join(args.summary_output_dir, 'summary_train.csv')
    summary_file_template = os.path.join(args.summary_output_dir, 'summary_template.csv')
    df_train = pd.DataFrame(columns=SUMMMARY_COLUMNS)
    df_template = pd.DataFrame(columns=SUMMMARY_COLUMNS)

    for real3d_class in constants.real3d_object_classes:
        logging.info(f'Reading Real3DAD class {real3d_class}')
        template_loader = DataLoader(
            real3d_dloaders.Dataset3DADTrain(args.real3dad_data_path, real3d_class, normalize=True),
            num_workers=num_workers, batch_size=1, shuffle=False, drop_last=False,
            collate_fn=util_dloaders.dloader_dict_collate)
        pbar = tqdm(enumerate(template_loader), total=len(template_loader))
        for idx, item in pbar:
            if item is None or not validate_item(SHAPENET_DS_NAME, item):
                continue
            assert(item['object_label'][0] == constants.GOOD_MASK)
            df_template = save_output_file(out_template_dir, REAL3D_DS_NAME, real3d_class, idx, item, df_template)
        df_template.to_csv(summary_file_template, index=False)
        train_loader = DataLoader(
            real3d_dloaders.Dataset3DADTest(args.real3dad_data_path, real3d_class, normalize=True),
            num_workers=num_workers, batch_size=1, shuffle=False, drop_last=False,
            collate_fn=util_dloaders.dloader_dict_collate)
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for idx, item in pbar:
            if item is None or not validate_item(SHAPENET_DS_NAME, item):
                continue
            df_train = save_output_file(out_train_dir, REAL3D_DS_NAME, real3d_class, idx, item, df_train)
        df_train.to_csv(summary_file_train, index=False)

    for mulsen_class in constants.mulsen_object_classes:
        num_good = 0
        num_anomalous = 0
        logging.info(f'Reading MulSen class {mulsen_class}')
        # First produce the "test" set, where we may have a mix of good and anomalous shapes. Count the number of
        # each shape type (good vs anomaly)
        train_loader = DataLoader(
            mulsen_dloaders.DatasetMulSenTest(args.mulsen_data_path, mulsen_class, normalize=True),
            num_workers=num_workers, batch_size=1, shuffle=False, drop_last=False,
            collate_fn=util_dloaders.dloader_dict_collate)
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for idx_train, item in pbar:
            if item is None or not validate_item(MULSEN_DS_NAME, item):
                # This entry is somehow problematic; skip it
                continue
            if item['object_label'][0] == constants.GOOD_MASK:
                num_good += 1.0
            else:
                num_anomalous += 1.0
            df_train = save_output_file(out_train_dir, MULSEN_DS_NAME, mulsen_class, idx_train, item, df_train)
        df_train.to_csv(summary_file_train, index=False)

        good_datapoints_needed = int(num_anomalous * TARGET_GOOD_TO_ANOMALY_PROPORTION - num_good)
        if good_datapoints_needed < 0:
            good_datapoints_needed = 0
        else:
            logging.info(f'May add {good_datapoints_needed} additional good datapoints into the training set')

        template_loader = DataLoader(
            mulsen_dloaders.DatasetMulSenTrain(args.mulsen_data_path, mulsen_class, normalize=True),
            num_workers=num_workers, batch_size=1, shuffle=False, drop_last=False,
            collate_fn=util_dloaders.dloader_dict_collate)
        pbar = tqdm(enumerate(template_loader), total=len(template_loader))
        for idx_template, item in pbar:
            if item is None or not validate_item(MULSEN_DS_NAME, item):
                # This entry is somehow problematic; skip it
                continue
            assert(item['object_label'][0] == constants.GOOD_MASK)
            if good_datapoints_needed > 0:
                # We need more datapoints into the train set into "good" category.
                good_datapoints_needed -= 1
                df_train = save_output_file(out_train_dir, MULSEN_DS_NAME, mulsen_class, idx_train, item, df_train)
                idx_train += 1
            else:
                df_template = save_output_file(
                    out_template_dir, MULSEN_DS_NAME, mulsen_class, idx_template, item, df_template)
        df_train.to_csv(summary_file_train, index=False)
        df_template.to_csv(summary_file_template, index=False)

    for ashapenet_class in constants.shapenet_object_classes:
        logging.info(f'Reading Anomaly-ShapeNet class {ashapenet_class}')
        template_loader = DataLoader(
            shapenet_dloaders.DatasetAnomalyShapenetTrain(args.anomalyshapenet_data_path, ashapenet_class, normalize=True),
            num_workers=num_workers, batch_size=1, shuffle=False, drop_last=False,
            collate_fn=util_dloaders.dloader_dict_collate)
        pbar = tqdm(enumerate(template_loader), total=len(template_loader))
        for idx, item in pbar:
            if item is None or not validate_item(SHAPENET_DS_NAME, item):
                continue
            assert (item['object_label'][0] == constants.GOOD_MASK)
            df_template = save_output_file(out_template_dir, SHAPENET_DS_NAME, ashapenet_class, idx, item,  df_template)
        df_template.to_csv(summary_file_template, index=False)
        train_loader = DataLoader(
            shapenet_dloaders.DatasetAnomalyShapenetTest(args.anomalyshapenet_data_path, ashapenet_class, normalize=True),
            num_workers=num_workers, batch_size=1, shuffle=False, drop_last=False,
            collate_fn=util_dloaders.dloader_dict_collate)
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for idx, item in pbar:
            if item is None or not validate_item(SHAPENET_DS_NAME, item):
                continue
            df_train = save_output_file(out_train_dir, SHAPENET_DS_NAME, ashapenet_class, idx, item, df_train)
        df_train.to_csv(summary_file_train, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real3dad_data_path", type=str, help='Path to the Real3DAD dataset')
    parser.add_argument("--mulsen_data_path", type=str, help='Path to the MulSet dataset')
    parser.add_argument("--anomalyshapenet_data_path", type=str, help='Path to the Anomaly-ShapeNet dataset')
    parser.add_argument("--output_data_path", type=str,
                        help='Path to the output folder, where the data will be saved to')
    parser.add_argument("--summary_output_dir", type=str,
                        help='Path to the output dir that summarizes the dataset')
    args = parser.parse_args()
    pipeline(args)
