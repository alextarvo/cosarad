""" Constants for the COSARAD project for industrial anomaly detection in point clouds"""
from collections import defaultdict
import numpy as np

# label values for the good and anomalous points, as well as for the whole PCs
GOOD_MASK = 0
ANOMALY_MASK = 1

# The textual description of non-anomalous object
NO_ANOMALY = 'good'

ALL_CLASSES_NAME = 'all_classes'
REAL3D_ALL_CLASSES_NAME = 'real3d_all_classes'
# Classes of the objects in Real3D dataset. Correspond to the subfolders in the real3d dataset
real3d_object_classes = [
    'shell','starfish', 'airplane', 'car','candybar', 'chicken',
    'diamond','duck','fish', 'gemstone', 'seahorse','toffees'
]


MULSEN_ALL_CLASSES_NAME = 'mulsen_all_classes'
# Classes of the objects in MulSen dataset.
mulsen_object_classes = [
    # 'cotton',  'zipper' are non-rigid objects and are exluded.
    # 'screw', 'spring_pad' have very noisy measurementns and also included for now

    'button_cell', 'capsule', 'cube', 'flat_pad',
    'light', 'nut',
    'piggy', 'plastic_cylinder', 'screen', 'solar_panel', 'toothbrush', ]

# mulsen_anomaly_free_types = ['good', 'color', 'broken_inside', 'detachment_inside']
mulsen_anomaly_free_types = ['good', 'color']

SHAPENET_ALL_CLASSES_NAME = 'shapenet_all_classes'
# Classes of the objects in Anomaly-ShapeNet dataset.
shapenet_object_classes = [
    'ashtray0', 'bag0', 'bottle0', 'bottle1', 'bottle3', 'bowl0', 'bowl1', 'bowl2', 'bowl3', 'bowl4', 'bowl5',
    'bucket0', 'bucket1', 'cabinet0', 'cap0', 'cap1', 'cap2', 'cap3', 'cap4', 'cap5', 'chair0', 'cup0', 'cup1', 'cup2',
    'desk0', 'eraser0', 'headset0', 'headset1', 'helmet0', 'helmet1', 'helmet2', 'helmet3',
    'jar0', 'knife0', 'knife1', 'microphone0', 'microphone1', 'screen0', 'shelf0', 'tap0', 'tap1',
    'vase0', 'vase1', 'vase10', 'vase2', 'vase3', 'vase4', 'vase5', 'vase6', 'vase7', 'vase8', 'vase9'
]

# If object -> template mean distance is over this threshold,  registration is unsuccessful
registration_mean_distances_thresholds = defaultdict(
    lambda: np.finfo(np.float32).max,
    {
        'shell': 0.1,
        'starfish': 0.2,
        'airplane': 0.5,
        'car': 1.0,
        'candybar': 0.12,
        'chicken': 0.35,
        'diamond': 0.2,
        'duck': 0.2,
        'fish': 0.1,
        'gemstone': 0.15,
        'seahorse': 0.1,
        'toffees': 0.1,

        # MulSen objects
        'light': 0.16,
        'nut': 0.12,
    }
)