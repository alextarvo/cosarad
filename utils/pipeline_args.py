import argparse

def set_pipeline_args():
    """
       Sets up command line arguments parsing for all our main pipelines
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--real3dad_data_path", type=str, help='Path to the Real3DAD dataset')
    parser.add_argument("--object_classes", type=str, default=None,
                        help='Name of the class to run the program upon')
    parser.add_argument("--do_visualize", action='store_true',
                        help='Visualize point clouds along with the registration template')
    parser.add_argument("--do_random_rotate", action='store_true',
                        help='Randomly rotate the test object')
    parser.add_argument("--do_registration", action='store_true',
                        help='Register point clouds')
    parser.add_argument("--obj_per_class", type=int, default=1000,
                        help='Process only a given number of objects per class. Used for debugging / visulization')
    parser.add_argument('--encoder', type=str, default='fpfh',
                        help='Type of an encoder to use to extract features. '
                        'Possible values: fpfh, pointnet, pointnet2_ssg, pointnet2_msg')
    parser.add_argument('--contrastive_encoder_path', type=str, default=None,
                        help='Type of an encoder to use to extract features. '
                        'Possible values: fpfh, pointnet, pointnet2_ssg, pointnet2_msg')
    parser.add_argument('--max_registration_attempts', type=int, default=5,
                        help='The number of tries to do the best registration for registration-based pipeline')
    parser.add_argument("--patch_radius", type=float, default=2.0,
                        help='The radius of a patch to be selected. If we are using a contrastive feature extractor, '
                        'it must be exactly the radius of the patch in the training set')
    # args = parser.parse_args()
    return parser
