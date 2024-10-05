# -*- coding: utf-8 -*-
# @Author: Philipp N. Mueller
# @Date:   2024-10-03 22:10:42
# @Last Modified by:   Philipp N. Mueller
# @Last Modified time: 2024-10-05 15:22:25


import trimesh
import numpy as np
import random
import matplotlib.pyplot as plt
import plotly.graph_objects as go


# Function to generate a synthetic tooth shape (an ellipsoid)
def generate_synthetic_tooth():
    # Create a random ellipsoid shape
    radii = [random.uniform(0.5, 1.0), random.uniform(0.8, 1.2), random.uniform(1.0, 1.5)]
    ellipsoid = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    
    # Scale the ellipsoid to simulate different tooth shapes
    ellipsoid.vertices *= radii
    
    return ellipsoid


def render_3d_tooth(tooth_model):
    # Extract vertices and faces
    vertices = tooth_model.vertices # Points defining the shape
    faces = tooth_model.faces # Surfaces connecting the points

    # Create a 3D plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the triangular faces
    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                    triangles=faces, color='lightblue', edgecolor='none', alpha=0.5)

    # Set plot limits (adjust as necessary)
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])

    # Set labels
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title("3D Plot of Synthetic Tooth Model")

    # Show the plot
    plt.show()



if __name__ == "__main__":
    ## Generate 100 teeth
    # Generate a larger collection of synthetic teeth (e.g., 100)
    # num_teeth = 100
    # synthetic_teeth = [generate_synthetic_tooth() for _ in range(num_teeth)]

    # # Visualize one synthetic tooth
    # synthetic_teeth[0].show()  # This will render a 3D view of the synthetic tooth

    # # To represent the collection in a simple way, let's calculate and store their volumes as a basic feature
    # tooth_volumes = [tooth.volume for tooth in synthetic_teeth]

    # # Show statistics of the generated tooth collection (e.g., average volume, min/max)
    # volume_stats = {
    #     "average_volume": np.mean(tooth_volumes),
    #     "min_volume": np.min(tooth_volumes),
    #     "max_volume": np.max(tooth_volumes)
    # }

    # print(volume_stats)  # Display basic stats about the collection of synthetic teeth
    
    # render_3d_tooth(synthetic_teeth[0])
    
    pass