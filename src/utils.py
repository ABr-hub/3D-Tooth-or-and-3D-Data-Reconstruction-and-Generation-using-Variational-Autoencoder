# -*- coding: utf-8 -*-
# @Author: ABr_hub
# @Date:   2024-10-05 11:30:01
# @Last Modified by:   ABr_hub
# @Last Modified time: 2024-10-05 15:54:53

import torch
from scipy.spatial import distance
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# Calculate chamfer distance between two point clouds
def chamfer_distance(pointcloud1, pointcloud2):
    dists1 = distance.cdist(pointcloud1, pointcloud2).min(axis=1)
    dists2 = distance.cdist(pointcloud2, pointcloud1).min(axis=1)
    return np.mean(dists1) + np.mean(dists2)

def visualize_latent_space(model, data):
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
