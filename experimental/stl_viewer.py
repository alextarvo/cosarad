import numpy as np
import open3d as o3d

# fname = '/mnt/data/MulSen/MulSen_AD/piggy/Pointcloud/train/0.stl'
# fname = '/mnt/data/MulSen/MulSen_AD/piggy/Pointcloud/test/broken/0.stl'
# fname = '/mnt/data/MulSen/MulSen_AD/piggy/Pointcloud/test/crack/0.stl'
# fname = '/mnt/data/MulSen/MulSen_AD/piggy/Pointcloud/test/color/0.stl'

fname = '/mnt/data/MulSen/MulSen_AD/spring_pad/Pointcloud/train/0.stl'

# fname = '/mnt/data/MulSen/MulSen_AD/nut/Pointcloud/test/broken/0.stl'
# fname = '/mnt/data/MulSen/MulSen_AD/nut/Pointcloud/train/0.stl'


#fname = '/mnt/data/Anomaly-ShapeNet-v2/dataset/obj/cabinet0/cabinet0_hole1.obj'
# fname = '/mnt/data/Anomaly-ShapeNet-v2/dataset/obj/vase0/vase0_scratch1.obj'
# fname = '/mnt/data/Anomaly-ShapeNet-v2/dataset/obj/vase1/vase1_positive4.obj'

# fname = '/mnt/data/MulSen/MulSen_AD/solar_panel/Pointcloud/test/good/2.stl'
# fname = '/mnt/data/MulSen/MulSen_AD/solar_panel/Pointcloud/test/broken/1.stl'
# fname = '/mnt/data/MulSen/MulSen_AD/solar_panel/Pointcloud/test/broken/2.stl'
# fname = '/mnt/data/MulSen/MulSen_AD/solar_panel/Pointcloud/test/broken/5.stl'
# fname = '/mnt/data/MulSen/MulSen_AD/solar_panel/Pointcloud/test/broken/6.stl'

# fname = '/mnt/data/MulSen/MulSen_AD/solar_panel/Pointcloud/test/foreign_body/0.stl'

fname = '/mnt/data/MulSen/MulSen_AD/toothbrush/Pointcloud/test/good/0.stl'
fname = '/mnt/data/MulSen/MulSen_AD/toothbrush/Pointcloud/test/good/1.stl'
fname = '/mnt/data/MulSen/MulSen_AD/toothbrush/Pointcloud/test/foreign_body/1.stl'
fname = '/mnt/data/MulSen/MulSen_AD/toothbrush/Pointcloud/test/scratch/2.stl'
fname = '/mnt/data/MulSen/MulSen_AD/toothbrush/Pointcloud/test/broken/0.stl'

fname = '/mnt/data/MulSen/MulSen_AD/button_cell/Pointcloud/test/hole/0.stl'
fname = '/mnt/data/MulSen/MulSen_AD/capsule/Pointcloud/test/broken_outside/1.stl'

fname = '/mnt/data/MulSen/MulSen_AD/capsule/Pointcloud/test/hole/5.stl'

fname = '/mnt/data/MulSen/MulSen_AD/cube/Pointcloud/test/broken/0.stl'

fname = '/mnt/data/MulSen/MulSen_AD/screen/Pointcloud/test/crack/6.stl'


def normalize_pc(point_cloud):
    """Centers the point cloud at 0,0.
    Works for actual PCs or for a PC where 4th column is an anomaly label
    """
    center = np.average(point_cloud[:, :3], axis=0)
    point_cloud[:, :3] -= center
    return point_cloud



# Load STL file as triangle mesh
mesh = o3d.io.read_triangle_mesh(fname)
mesh.compute_vertex_normals()

# Visualize
o3d.visualization.draw_geometries([mesh])

pcd = mesh.sample_points_uniformly(number_of_points=100000)
o3d.visualization.draw_geometries([pcd])
np_pcd = normalize_pc(np.array(pcd.points))
print(np_pcd.min(axis=0))
print(np_pcd.max(axis=0))
