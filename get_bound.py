import open3d as o3d

ply_path = '/disk/lsf/lsf_datasets/new_college/pcd/1583840263.540367616.ply'
point_cloud = o3d.io.read_point_cloud(ply_path)

aabb = point_cloud.get_axis_aligned_bounding_box()

min_corner = aabb.get_min_bound()
max_corner = aabb.get_max_bound()

print("Min corner:", min_corner)
print("Max corner:", max_corner)
