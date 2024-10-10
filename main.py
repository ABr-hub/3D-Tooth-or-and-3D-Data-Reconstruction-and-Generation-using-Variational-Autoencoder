# -*- coding: utf-8 -*-
# @Author: ABr_hub
# @Date:   2024-10-03 22:18:53
# @Last Modified by:   ABr_hub
# @Last Modified time: 2024-10-10 21:27:13


import os

from src.syntheticTeeth import *
from src.preprocessingTeeth import *
from src.utils import *
from src.vae import *

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# NOTE: This is necessary because multiple instances of OpenMP runtime are loaded
# eg. through trimesh or pytorch; This solution can lead to unstable results.
# A better solution would be to asure that only one instance of the OpenMP runtime
# is loaded. Herefore e.g. the libraries using that runtime could to be updated 
# or switched to versions that dont use that linking
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def data_generator(
    n_teeth: int=100,
    split: float=0.8,
    batch_size: int=16,
    net: str='linear'
) -> tuple:
    
    # Generate a few synthetic teeth
    synthetic_teeth = [generate_synthetic_tooth() for _ in range(n_teeth)]
    
    # To represent the collection in a simple way, let's calculate and store their volumes as a basic feature
    tooth_volumes = [tooth.volume for tooth in synthetic_teeth]

    # Show statistics of the generated tooth collection (e.g., average volume, min/max)
    volume_stats = {
        "average_volume": np.mean(tooth_volumes),
        "min_volume": np.min(tooth_volumes),
        "max_volume": np.max(tooth_volumes)
    }
    
    print(volume_stats)  # Display basic stats about the collection of synthetic teeth


    # Convert all synthetic teeth into point clouds
    synthetic_pointclouds = np.array([
        mesh_to_pointcloud(tooth) for tooth in synthetic_teeth
    ])
    
    
    # convert data into respective form and shape
    if net == 'cnn':
        synthetic_voxel_grid = pointcloud_to_voxel_grid(synthetic_pointclouds)
        
        # Convert data to pytorch tensors
        synthetic_voxel_grid_tensors =  torch.tensor(
            synthetic_voxel_grid, dtype=torch.float32
        )
        
        # reshape to be used in 3d layer (B, C, H, W, D)
        synthetic_voxel_grid_tensors = synthetic_voxel_grid_tensors.unsqueeze(1)
        
        synthetic_data = synthetic_voxel_grid_tensors
        
    else:
        # Convert data to pytorch tensors
        synthetic_pointclouds_tensors =  torch.tensor(
            synthetic_pointclouds, dtype=torch.float32
        )
        
        # flatten the tensor dimension since here only a simple mlp is defined
        synthetic_pointclouds_tensors = synthetic_pointclouds_tensors.view(
            synthetic_pointclouds_tensors.size(0), -1
        )
        
        synthetic_data = synthetic_pointclouds_tensors

    
    # split data in defined ratio
    msk = np.random.rand(synthetic_data.shape[0]) < split
    train_data = synthetic_data[msk]
    test_data = synthetic_data[~msk]
    
    print('')
    print('train_data.shape', train_data.shape)
    print('test_data.shape', test_data.shape)

    
    # Convert into pytorch datasets
    train_data = TensorDataset(train_data)
    test_data = TensorDataset(test_data)
    
    # Translate to dataloader objects
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=test_data.tensors[0].shape[0])
    
    return train_loader, test_loader, synthetic_data


def reconstruct_data(
    model, 
    original_data: torch.tensor, 
    idx,
    plot: True,
    vae_type='linear'
):
    # Get model from GPU
    model = model.to("cpu")
    
    # Switch to eval mode 
    model.eval()    
    
    if vae_type == 'linear':
        # Flat representation of the data due to the network structure
        pointcloud_data = original_data[idx].view(-1, 1024 * 3)
    else:
        pointcloud_data = original_data[idx].unsqueeze(1)
    
    with torch.no_grad():
        # Reconstruct tooth
        recon_data, _, _ = model(pointcloud_data)
        
        if vae_type == 'linear':
            # Transform back to 3D data
            recon_data = recon_data.view(-1, 3).numpy()
    
    # Visualize the oiginal & reconstructed tooth
    original_data_points = original_data[idx].numpy()
    
    # Visualize reconstruction
    if plot == True:
        visualize_pointcloud(original_data_points, title="Original Data")
        visualize_pointcloud(recon_data, title="Reconstructed Data")
        
    return recon_data
    
    
def generate_new_data(
    model, 
    plot: True,
    vae_type='linear'
):
    # Switch to eval mode 
    model.eval()
    
    with torch.no_grad():
        # Create random point in latent space
        z = torch.randn(1, model.latent_dim)
        generated_data = model.decode(z)
        
        if vae_type == 'linear':
            generated_data = generated_data.view(-1, 3).numpy()
        
    # Visualize generated data
    if plot == True:
        visualize_pointcloud(generated_data, title="Generated Data")
    
    return generated_data
    

def main(vae_type='linear'):
    # Get training and test data and one
    train_loader, test_loader, original_point_clouds = data_generator(n_teeth=100, net=vae_type)  

    # get model
    vae, _ = create_vae_net(latent_dim=16, vae_type=vae_type)
    
    # Test model for functionality
    if vae_type == 'linear':
        X = next(iter(train_loader))[0] 
        X = X.view(X.size(0), -1)
        recon_x, mean, logvar = vae(X)
    else: 
        X = next(iter(train_loader))[0] 
        recon_x, mean, logvar = vae(X)
    
    print('\nValues: ', recon_x, mean, logvar)
    print('\nShapes vae: ', recon_x.shape, mean.shape, logvar.shape)
    print('Shapes X: ', X.shape)
    print('')


    # train the model
    trainLoss, testLoss, vae = train_vae(
        train_loader=train_loader,
        dev_loader=test_loader,
        epochs=75,
        latent_dim=32,
        vae_type=vae_type
    )
    
    # visualize training history
    fig, axs = plt.subplots(1,1, figsize=(6,3))

    axs.plot(trainLoss, label='train_loss')
    axs.plot(testLoss, label='test_loss')
    axs.set_xlabel('Epochs')
    axs.set_ylabel('Loss')
    axs.legend()
    axs.grid()

    plt.show()

    
    # Reconstruct tooth data
    reconstructed_tooth = reconstruct_data(vae, original_point_clouds, 5, False, vae_type=vae_type)
    
    # Generate new tooth data
    generated_tooth = generate_new_data(vae, False, vae_type=vae_type)
    

    # Calculate Chamfer Distance as evaluation metric
    if vae_type == 'linear':
        original_points = original_point_clouds[0].view(1024, 3).numpy() # expected shape (1024, 3)
        reconstructed_points = reconstructed_tooth
        print(' ')
        print(f"Chamfer Distance: {chamfer_distance(original_points, reconstructed_points)}")

        # Visualize results
        fig = plt.figure(figsize=(20,10))
        
        # Original vs. reconstruction
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(original_points[:, 0], original_points[:, 1], original_points[:, 2], color='b', label='Original')
        ax1.scatter(reconstructed_points[:, 0], reconstructed_points[:, 1], reconstructed_points[:, 2], color='r', label='Reconstructed')
        ax1.set_xlabel('X-Axis')
        ax1.set_ylabel('Y-Axis')
        ax1.set_zlabel('Z-Axis')
        ax1.legend()
        
        # Original vs. generated
        ax1 = fig.add_subplot(122, projection='3d')
        ax1.scatter(original_points[:, 0], original_points[:, 1], original_points[:, 2], color='b', label='Original')
        ax1.scatter(generated_tooth[:, 0], generated_tooth[:, 1], generated_tooth[:, 2], color='r', label='Generated')
        ax1.set_xlabel('X-Axis')
        ax1.set_ylabel('Y-Axis')
        ax1.set_zlabel('Z-Axis')
        ax1.legend()

        # Render plot
        plt.show()
        
    # transform 3d voxel grid to point cloud to compute chamfer_distance
    else:
        #original_points = voxel_grid_to_pointcloud(original_point_clouds[0])
        #reconstructed_points = voxel_grid_to_pointcloud(reconstructed_tooth)

        # Create a 3D Boolean mask for the voxels (True = voxel present, False = empty)
        voxel_bool_org = torch.squeeze(original_point_clouds[0]) > 0      # True for values > 0 (voxel occupied)
        voxel_bool_recon = torch.squeeze(reconstructed_tooth) > 0                      # True for values > 0 (voxel occupied)
        
        
        # Visualize results
        fig = plt.figure(figsize=(20,10))
        
        # Original vs. reconstruction
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.voxels(voxel_bool_org, facecolors='blue', edgecolor='k')
        
        
        # Original vs. generated
        ax1 = fig.add_subplot(122, projection='3d')
        ax1.voxels(voxel_bool_recon, facecolors='blue', edgecolor='k')
        
        plt.show()
        
    
    # Visualize latent space
    # ... 
    

if __name__ == "__main__":
    main('cnn')
