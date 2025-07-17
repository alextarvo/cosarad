import logging

import open3d as o3d
import numpy as np

from . import util_registration as util_reg
from . import base_registration as base_reg


class RegistrationICP(base_reg.RegistrationBase):
    """Uses ICP algorithm to register ppoint clouds"""
    def __init__(self, np_pc_template=None, np_pc_target=None, voxel_size=0.5, icp_distance_threshold=0.4):
        super(RegistrationICP, self).__init__(np_pc_template, np_pc_target)
        self.voxel_size = voxel_size
        self.icp_distance_threshold = icp_distance_threshold
        self.result_ransac = None
        self.result_icp = None

    def _preprocess_point_cloud(self, o3d_pc):
        """downsample the point cloud, compute its normals and FPFH descriptors."""
        # print(":: Downsample with a voxel size %.3f." % voxel_size)
        pcd_downsampled = o3d_pc.voxel_down_sample(self.voxel_size)
        logging.info(f'Dwnsampled the PC with voxel {self.voxel_size} from {len(o3d_pc.points)} '
                     f'to {len(pcd_downsampled.points)} points')

        radius_normal = self.voxel_size * 2
        pcd_downsampled.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = self.voxel_size * 5
        # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_downsampled, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        return pcd_downsampled, pcd_fpfh

    def _prepare_dataset(self):
        """Convert the source and target point clouds from Numpy 3D arrays into the Open3D unorganized point cloud.
        During the pre-processing, downsample the point cloud, compute its normals and FPFH descriptors.

        Returns:
            for each object, return its unorganized PC in original resolution; a downsampled one; and its FPFH descriptors
        """
        o3d_pc_template = o3d.geometry.PointCloud()
        o3d_pc_template.points = o3d.utility.Vector3dVector(self.np_pc_template)
        o3d_pc_target = o3d.geometry.PointCloud()
        o3d_pc_target.points = o3d.utility.Vector3dVector(self.np_pc_target)

        # alexta: This seems to be purely for testing purposees  - rotate source point cloud
        # summary_pointcloud(target)
        # source = random_rotate(source)
        # draw_registration_result(source, target, np.identity(4))

        o3d_pc_template_down, template_fpfh = self._preprocess_point_cloud(o3d_pc_template)
        o3d_pc_target_down, target_fpfh = self._preprocess_point_cloud(o3d_pc_target)
        return o3d_pc_template, o3d_pc_target, o3d_pc_template_down, o3d_pc_target_down, template_fpfh, target_fpfh

    def _execute_global_registration(self, o3d_pc_target_down, o3d_pc_template_down, source_fpfh, template_fpfh):
        """ Do an _initial_ global registration of point clouds using global RANSAC-based registration"""
        # distance_threshold = voxel_size * 1.5
        distance_threshold = self.voxel_size * 1.2
        # print(":: RANSAC registration on downsampled point clouds.")
        # print("   Since the downsampling voxel size is %.3f," % voxel_size)
        # print("   we use a liberal distance threshold %.3f." % distance_threshold)
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            o3d_pc_target_down, o3d_pc_template_down, source_fpfh, template_fpfh, True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                    0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
        return result

    def compute_registration_transform(self):
        o3d_pc_template, o3d_pc_target, o3d_pc_template_down, o3d_pc_target_down, template_fpfh, target_fpfh = (
            self._prepare_dataset())
        result_ransac = self._execute_global_registration(
            o3d_pc_target_down, o3d_pc_template_down, target_fpfh, template_fpfh)
        o3d_pc_target.transform(result_ransac.transformation)
        result_icp = o3d.pipelines.registration.registration_icp(
            o3d_pc_target, o3d_pc_template, self.icp_distance_threshold)
        o3d_pc_target.transform(result_icp.transformation)
        #    draw_registration_result(source, target, np.identity(4))
        self.np_pc_target_registered = np.asarray(o3d_pc_target.points)
        self.accuracy = util_reg.RegistrationAccuracy(o3d_pc_target, o3d_pc_template)
        self.result_ransac = result_ransac
        self.result_icp = result_icp

    def get_transform(self):
        return self.result_icp.transformation @ self.result_ransac.transformation
