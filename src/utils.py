# -*- coding: utf-8 -*-
# @Author: ABr_hub
# @Date:   2024-10-05 11:30:01
# @Last Modified by:   ABr_hub
# @Last Modified time: 2024-10-10 15:49:54

import torch
from scipy.spatial import distance
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# Calculate chamfer distance between two point clouds
def chamfer_distance(
    pointcloud1, 
    pointcloud2
) -> np.ndarray:
    '''
    The Chamfer Distance is a metric commonly used to compare two sets of points, 
    particularly in the context of point cloud data. It measures how similar or 
    different two point clouds are, and it's often used in 3D reconstruction 
    tasks (like in a Variational Autoencoder setup) to assess how well a 
    generated point cloud (output) matches the original one (input).
    '''
    dists1 = distance.cdist(pointcloud1, pointcloud2).min(axis=1)
    dists2 = distance.cdist(pointcloud2, pointcloud1).min(axis=1)
    
    return np.mean(dists1) + np.mean(dists2)


def voxel_grid_to_pointcloud(voxel_grid):
    '''
    Converts a voxel grid of shape (32, 32, 32) into a point cloud.
    Only the voxels with value > 0 are considered as points in the point cloud.
    '''
    # Find the indices where the value is greater than 0 (or occupied voxel positions)
    points = np.argwhere(voxel_grid > 0)
    return points


def visualize_latent_space(
    model, 
    data
):
    with torch.no_grad():
        latent_representations = model.get_latent_representation(data)
    
    tsne = TSNE(n_components=2)
    reduced_data = tsne.fit_transform(latent_representations.numpy())

    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], alpha=0.5)
    plt.title('Latent Space Visualization')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.show()


if __name__ == "__main__":
    pass
