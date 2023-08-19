import torch
def compute_iou(point_cloud1, point_cloud2, threshold):
    # Calculate pairwise distances between points in point_cloud1 and point_cloud2
    pairwise_distances = torch.cdist(point_cloud1, point_cloud2)

    # Find the minimum distances between each point in point_cloud1 and point_cloud2
    min_distances_1 = torch.min(pairwise_distances, dim=1).values
    min_distances_2 = torch.min(pairwise_distances, dim=0).values

    # Calculate the number of points in each point cloud
    num_points_cloud1 = point_cloud1.shape[0]
    num_points_cloud2 = point_cloud2.shape[0]

    # Calculate the number of overlapping points (IoU numerator)
    num_overlapping_points_1 = torch.sum(min_distances_1 < threshold).item()
    num_overlapping_points_2 = torch.sum(min_distances_2 < threshold).item()

    # find overlapping points in point_cloud1
    overlapping_points1 = point_cloud1[min_distances_1 < threshold]

    # find overlapping points in point_cloud2
    overlapping_points2 = point_cloud2[min_distances_2 < threshold]

    # Calculate the total number of points in both point clouds (IoU denominator)
    total_points = num_points_cloud1 + num_points_cloud2

    # Calculate the IoU
    iou = (num_overlapping_points_1 + num_overlapping_points_2) / float(total_points)

    overlapping_points1 = overlapping_points1.numpy()
    overlapping_points2 = overlapping_points2.numpy()


    return iou, overlapping_points1, overlapping_points2
