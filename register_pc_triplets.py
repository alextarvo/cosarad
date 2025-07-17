"""Registers PC triplets - template (good), good, anomalous - into a single frame of reference."""
import os
import logging
import pandas as pd
import dataloaders.cosarad_dataloaders as cosarad_dloaders
import dataloaders.util_dataloaders as util_dloaders
import registration.icp_registration as icp_reg
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.debug import in_debugger

import constants as consts

import argparse
import random

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)
logging.basicConfig(level=logging.INFO)

num_workers = 0 if in_debugger() else 4


# This is for testing
# random.seed(42)

def get_args():
    """
       Sets up command line arguments parsing
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--cosarad_data_path", type=str, help='Path to the Real3DAD dataset')
    parser.add_argument("--object_classes", type=str, default=None,
                        help='Name of the class to run the program upon')
    parser.add_argument("--num_objects", type=int, default=100,
                        help='Process only a given number of objects per class. Used for debugging / visulization')
    parser.add_argument("--registration_results_output_path", type=str,
                        help='Path where to save the registration results')
    parser.add_argument("--top_n", type=int, default=5,
                        help='Number of best matches to keep for each bad case')
    return parser.parse_args()


def collect_registration_data(args, object_class):
    """For each anomalous object in the dataset, collect register it against two "good" objects.
    Pick top_n registrations, and save them into the resulting dataframe.

    Args:
        args: command line arguments
        object_class: Object class which should be registered
    """
    data = []

    good_dataset = cosarad_dloaders.DatasetCosarad(
        args.cosarad_data_path, 'train', dataset_tpl='.*', class_tpl=object_class,
        anomaly_tpl='good', index_tpl='.*', normalize=False)
    bad_dataset = cosarad_dloaders.DatasetCosarad(
        args.cosarad_data_path, 'train', dataset_tpl='.*', class_tpl=object_class,
        anomaly_tpl='(?!good)\w+', index_tpl='.*', normalize=False)
    assert len(bad_dataset) > 0, f'No anomalous files were detected for the object class {object_class}'
    assert len(good_dataset) > 0, f'No good files were detected for the object class {object_class}'
    # Indices of good PCs for this object
    good_indices = [i for i in range(len(good_dataset))]
    bad_loader = DataLoader(bad_dataset, num_workers=num_workers, batch_size=1, shuffle=False, drop_last=False,
                            collate_fn=util_dloaders.dloader_dict_collate)
    pbar = tqdm(enumerate(bad_loader), total=len(bad_loader))
    logging.info(f'Registering PC triplets for class {object_class}')
    for idx_bad, item_bad in pbar:
        bad_case_results = []
        if item_bad['np_pointcloud'] is None:
            logging.warning(f'Failed to retrieve a point cloud for a bad object {object_class}, entry no. {idx_bad}')
            continue
        assert len(
            item_bad['np_pointcloud']) == 1, f'We expect a single entry read from the {object_class}, {idx_bad} entry!'
        for template_idx in good_indices:
            # For a given bad PC, get two good ones: "anchor" and "good"
            other_good_indices = [i for i in good_indices if i != template_idx]
            assert other_good_indices, f'Failed to create a good entries index for {object_class}, bad entry no. {idx_bad}'
            good_idx = random.choice(other_good_indices)
            item_template = good_dataset[template_idx]
            item_good = good_dataset[good_idx]
            reg_good = icp_reg.RegistrationICP(item_template['np_pointcloud'], item_good['np_pointcloud'])
            reg_bad = icp_reg.RegistrationICP(item_template['np_pointcloud'], item_bad['np_pointcloud'][0])

            # Register good PC against the template
            reg_good.compute_registration_transform()
            # If mean distance is too large, skip this template
            if reg_good.accuracy.mean_dist_src_to_target > consts.registration_mean_distances_thresholds[object_class]:
                continue
            # Register anomalous PC against the template
            reg_bad.compute_registration_transform()
            if reg_bad.accuracy.mean_dist_src_to_target > consts.registration_mean_distances_thresholds[object_class]:
                continue
            good_transform = reg_good.get_transform()
            bad_transform = reg_bad.get_transform()

            # Record data for this template
            bad_case_results.append({
                'object_class': object_class,
                'template_file': os.path.basename(item_template['npz_file']),
                'good_file': os.path.basename(item_good['npz_file']),
                'bad_file': os.path.basename(item_bad['npz_file'][0]),
                'good_good_mean_dist': reg_good.accuracy.mean_dist_src_to_target,
                'good_good_max_dist': reg_good.accuracy.max_dist_src_to_target,
                'good_bad_mean_dist': reg_bad.accuracy.mean_dist_src_to_target,
                'good_bad_max_dist': reg_bad.accuracy.max_dist_src_to_target,
                'good_good_chamfer': reg_good.accuracy.chamfer_distance,
                'good_bad_chamfer': reg_bad.accuracy.chamfer_distance,
                'good_good_transform': util_dloaders.matrix_to_string(good_transform),
                'good_bad_transform': util_dloaders.matrix_to_string(bad_transform)
            })

        # Sort results by good_bad_mean_dist and keep top_n
        if bad_case_results:
            bad_case_results.sort(key=lambda x: x['good_bad_mean_dist'])
            data.extend(bad_case_results[:args.top_n])

    return pd.DataFrame(data)


if __name__ == "__main__":
    args = get_args()
    object_classes = util_dloaders.get_object_classes_names(args.object_classes)
    for object_class in object_classes:
        # Collect data for a given object class
        df = collect_registration_data(args, object_class)
        if len(df) > 0:
            # Save to CSV
            output_file = f"registration_data_{object_class}.csv"
            os.makedirs(args.registration_results_output_path, exist_ok=True)
            output_full_path = os.path.join(args.registration_results_output_path, output_file)
            df.to_csv(output_full_path, index=False)
            print(f"Data saved to {output_file}")
        else:
            print("No data collected")
