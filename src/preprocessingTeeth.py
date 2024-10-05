# -*- coding: utf-8 -*-
# @Author: ABr_hub
# @Date:   2024-10-03 22:18:23
# @Last Modified by:   ABr_hub
# @Last Modified time: 2024-10-05 15:55:06


import trimesh
import matplotlib.pyplot as plt


# Function to convert a synthetic tooth (mesh) into a point cloud
def mesh_to_pointcloud(tooth_mesh, num_points=1024):
    # Sample points on the surface of the tooth mesh
    points, _ = trimesh.sample.sample_surface(tooth_mesh, num_points)
    return points


def visualize_pointcloud(points, title="", figsize=(8, 6)):
    # Assuming points is a Nx3 array
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Plotting the point cloud
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='b')  
    ax.set_title(title)
    plt.show()


if __name__ == "__main__":
    pass
