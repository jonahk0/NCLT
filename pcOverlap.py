import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import read_vel_sync
from Iou import compute_iou

# bin file numbers and corresponding ground truth data, [x, y, z, roll, pitch, yaw]
#bin number is the time (number of microseconds after Thursday, 1 January 1970)
#0 s  1326030975726043
#30 s 1326031005526708
#60 s 1326031035527367
#19.075 m 1326031029927243 [4.610984706571884, -18.200168610833352, 1.495024063063006, -0.030486444816253, -0.031498646207296, -1.291138351522954]
#75 s 1326031050727693 [-7.613315562571650, -38.907718414503421, 2.284246181778894, 0.006079653881201, 0.039835909890419, 3.121846741302069]
#125 s 1326031100728781 [-54.070774542916368, -70.758247719545182, 2.368061431465930, 0.003022298663726, -0.017017062471257, -2.067790808387384]
#484 s 1326031459936062 [-87.143906287595357, -260.000037961159251, 7.375372588273502, -0.036776233911278, 0.009569662427886, 3.121480507917382]
# Generate the first point cloud
pose1 = [0.348864473789069, 0.22425948219136, 0.585903834704165, -0.065622381665258, -0.01985665979173, -0.024562199113143]
pose2 = [4.610984706571884, -18.200168610833352, 1.495024063063006, -0.030486444816253, -0.031498646207296, -1.291138351522954]
point_cloud_1 = read_vel_sync.main('/media/jonah/UBUSSD/Research/NYU/PCR/NCLT/2012-01-08/velodyne_sync/1326030975726043.bin', pose1)

point_cloud_2 = read_vel_sync.main("/media/jonah/UBUSSD/Research/NYU/PCR/NCLT/2012-01-08/velodyne_sync/1326031029927243.bin", pose2)

# Convert numpy arrays to Open3D.o3d.geometry.PointCloud
pcd1 = o3d.geometry.PointCloud()
pcd2 = o3d.geometry.PointCloud()
pcd1.points = o3d.utility.Vector3dVector(point_cloud_1)
pcd2.points = o3d.utility.Vector3dVector(point_cloud_2)

# Perform ICP
trans_init = np.asarray([[1, 0, 0, 0], 
                         [0, 1, 0, 0], 
                         [0, 0, 1, 0], 
                         [0, 0, 0, 1]])

reg_p2p = o3d.pipelines.registration.registration_icp(
    pcd1, pcd2, 0.5, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint())

# Transform second point cloud
pcd2.transform(reg_p2p.transformation)
point_cloud_2 = np.asarray(pcd2.points)

tensor_point_cloud1 = torch.tensor(point_cloud_1)
tensor_point_cloud2 = torch.tensor(point_cloud_2)

threshold = 10  # adjust this value as needed
iou, overlapping_points1, overlapping_points2 = compute_iou(tensor_point_cloud1, tensor_point_cloud2, threshold)
print(f"IoU between point clouds: {iou}")


# Plotting with matplotlib
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# First point cloud in blue with black edge color, second in red with black edge color
ax.scatter(point_cloud_1[:, 0], point_cloud_1[:, 1], -point_cloud_1[:, 2], c='b', s=5)
ax.scatter(point_cloud_2[:, 0], point_cloud_2[:, 1], -point_cloud_2[:, 2], c='r', s=5)

# Optional: Plot overlapping points from both point clouds in green
ax.scatter(overlapping_points1[:, 0], overlapping_points1[:, 1], -overlapping_points1[:, 2], c='g', s=5, edgecolors='k')
ax.scatter(overlapping_points2[:, 0], overlapping_points2[:, 1], -overlapping_points2[:, 2], c='g', s=5, edgecolors='k')

plt.show()