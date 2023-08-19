# !/usr/bin/python
#
# Example code to read a velodyne_sync/[utime].bin file
# Plots the point cloud using matplotlib. Also converts
# to a CSV if desired.
#
# To call:
#
#   python read_vel_sync.py velodyne.bin [out.csv]
#

import sys
import struct
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
import numpy as np

def convert(x_s, y_s, z_s):

    scaling = 0.005 # 5 mm
    offset = -100.0

    x = x_s * scaling + offset
    y = y_s * scaling + offset
    z = z_s * scaling + offset

    return x, y, z

def pose_to_transformation_matrix(pose):
    tx, ty, tz, roll, pitch, yaw = pose

    # Define rotation matrix using Euler angles
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    R = np.dot(Rz, np.dot(Ry, Rx))

    # Define translation vector
    t = np.array([tx, ty, tz]).reshape((3, 1))

    # Combine R and t to a 4x4 transformation matrix
    T = np.block([
        [R, t],
        [0, 0, 0, 1]
    ])

    return T

def transform_points(points, pose):
    # Convert the pose to a transformation matrix
    T = pose_to_transformation_matrix(pose)

    # Homogeneous coordinates
    points_h = np.hstack((points, np.ones((points.shape[0], 1))))

    # Apply transformation
    points_transformed_h = np.dot(T, points_h.T).T

    # Back to 3D coordinates
    points_transformed = points_transformed_h[:, :3]

    return points_transformed

def main(bin_file_path, pose):
    f_bin = open(bin_file_path, "rb")

    hits = []

    while True:

        x_str = f_bin.read(2)
        if len(x_str) < 2: # eof or data is not of expected length
            break

        y_str = f_bin.read(2)
        if len(y_str) < 2: # eof or data is not of expected length
            break

        z_str = f_bin.read(2)
        if len(z_str) < 2: # eof or data is not of expected length
            break

        i_str = f_bin.read(1)
        if len(i_str) < 1: # eof or data is not of expected length
            break

        l_str = f_bin.read(1)
        if len(l_str) < 1: # eof or data is not of expected length
            break

        x = struct.unpack('<H', x_str)[0]
        y = struct.unpack('<H', y_str)[0]
        z = struct.unpack('<H', z_str)[0]
        i = struct.unpack('B', i_str)[0]
        l = struct.unpack('B', l_str)[0]

        x, y, z = convert(x, y, z)

        s = "%5.3f, %5.3f, %5.3f, %d, %d" % (x, y, z, i, l)

        hits += [[x, y, z]]

    f_bin.close()
    hits = np.asarray(hits)
    hits = transform_points(hits, pose)

    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(hits[:, 0], hits[:, 1], -hits[:, 2], c='blue', s=5, linewidths=0) # first point cloud in blue

    #plt.show()

    return hits


if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))