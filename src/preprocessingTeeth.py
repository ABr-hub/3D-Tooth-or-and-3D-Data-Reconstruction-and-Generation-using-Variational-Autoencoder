# -*- coding: utf-8 -*-
# @Author: ABr_hub
# @Date:   2024-10-03 22:18:23
# @Last Modified by:   ABr_hub
# @Last Modified time: 2024-10-09 22:07:34


import trimesh
import matplotlib.pyplot as plt
import numpy as np


# Function to convert a synthetic tooth (mesh) into a point cloud
def mesh_to_pointcloud(tooth_mesh, num_points=1024):
    # Sample points on the surface of the tooth mesh
    points, _ = trimesh.sample.sample_surface(tooth_mesh, num_points)
    return points


def pointcloud_to_voxel_grid(pointclouds, grid_size=32):
    """
    Convert a numpy-array of point clouds into a 3d-voxel grid.

    Parameters
    ----------
    pointclouds (numpy array): Array of the form (B, N, 3), where B is the number 
        of point clouds, N is the number of points and 3 are the x, y, z coordinates.
    grid_size (int): The size of the voxel grid along each dimension (default is 
        32x32x32).
    
    Returns
    -------
    voxel_grids (numpy array): Voxel grids of the form (B, grid_size, grid_size, 
        grid_size) for each point cloud.
    """
    
    B, _, _ = pointclouds.shape  # B: Number of point clouds, N: Number of points, 3: Coordinates (B, N, 3)
    voxel_grids = np.zeros((B, grid_size, grid_size, grid_size), dtype=np.float32)  # (B, 32, 32, 32)
    
    # IIterate over each point cloud in the batch dimension
    for i in range(B):
        pointcloud = pointclouds[i]  # Single point cloud, shape: (N, 3)
        
        # Normalize the points to the range [0, 1)
        normalized_points = (pointcloud - np.min(pointcloud, axis=0)) / (np.ptp(pointcloud, axis=0) + 1e-5)
        # Normalized point cloud, shape: (N, 3)
        
        # Scale normalized points into voxel grid [0, grid_size)
        voxel_indices = np.floor(normalized_points * (grid_size - 1)).astype(int)
        # Voxel-Indizes, Form: 
        
        # Set the corresponding voxels 
        for voxel in voxel_indices:
            voxel_grids[i, voxel[0], voxel[1], voxel[2]] = 1.0
        # Voxel grid after processing the point cloud, Shape: (grid_size, grid_size, grid_size)1)
        

    return voxel_grids  # Return shape: (B, grid_size, grid_size, grid_size)


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
