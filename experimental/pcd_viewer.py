import open3d as o3d

fname = '/mnt/data/Anomaly-ShapeNet-v2/dataset/new_pcd/cabinet0/test/cabinet0_bulge1.pcd'
# fname = '/mnt/data/Anomaly-ShapeNet-v2/dataset/new_pcd/cabinet0/train/cabinet0_template0.pcd'

#fname = '/mnt/data/MulSen/MulSen_AD/piggy/Pointcloud/test/color/0.stl'

pcd = o3d.io.read_point_cloud(fname)
#o3d.visualization.draw_geometries([pcd])

print(len(pcd.points))