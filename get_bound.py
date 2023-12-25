import open3d as o3d

# 读取PLY文件
ply_path = '/disk/lsf/lsf_datasets/new_college/pcd/1583840263.540367616.ply'
point_cloud = o3d.io.read_point_cloud(ply_path)

# 计算包围框
aabb = point_cloud.get_axis_aligned_bounding_box()

# 获取包围框的最小角点和最大角点
min_corner = aabb.get_min_bound()
max_corner = aabb.get_max_bound()

# 输出包围框的最小角点和最大角点
print("Min corner:", min_corner)
print("Max corner:", max_corner)