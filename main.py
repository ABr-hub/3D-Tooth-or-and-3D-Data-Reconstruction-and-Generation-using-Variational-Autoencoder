# -*- coding: utf-8 -*-
# @Author: ABr_hub
# @Date:   2024-10-03 22:18:53
# @Last Modified by:   ABr_hub
# @Last Modified time: 2024-10-05 15:55:17


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
    n_teeth=100,
    split=0.8,
    batch_size=16
):
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
        
    # Convert data to pytorch tensors
    synthetic_pointclouds_tensors =  torch.tensor(
        synthetic_pointclouds, dtype=torch.float32
    )
    
    # split data in defined ratio
    msk = np.random.rand(synthetic_pointclouds_tensors.shape[0]) < split
    train_data = synthetic_pointclouds_tensors[msk]
    test_data = synthetic_pointclouds_tensors[~msk]
    
    print('')
    print('train_data.shape', train_data.shape)
    print('test_data.shape', test_data.shape)
    
    # flatten the tensor dimension since here only a simple mlp is defined
    train_data = train_data.view(train_data.size(0), -1)
    test_data = test_data.view(test_data.size(0), -1)
    
    # Convert into pytorch datasets
    train_data = TensorDataset(train_data)
    test_data = TensorDataset(test_data)
    
    # Translate to dataloader objects
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=test_data.tensors[0].shape[0])
    
    return train_loader, test_loader, synthetic_pointclouds_tensors


def reconstruct_data(
    model, 
    original_data: torch.tensor, 
    idx,
    plot: True
):
    # Get model from GPU
    model = model.to("cpu")
    
    # Switch to eval mode 
    model.eval()    
    
    # Flat representation of the data due to the network structure
    pointcloud_data = original_data[idx].view(-1, 1024 * 3)
    
    with torch.no_grad():
        # Reconstruct tooth
        recon_data, _, _ = model(pointcloud_data)
        
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
    plot: True
):
    # Switch to eval mode 
    model.eval()
    
    with torch.no_grad():
        # Create random point in latent space
        z = torch.randn(1, model.latent_dim)
        generated_data = model.decode(z)
        generated_data = generated_data.view(-1, 3).numpy()
        
    # Visualize generated data
    if plot == True:
        visualize_pointcloud(generated_data, title="Generated Data")
    
    return generated_data
    

def main():
    # Get training and test data and one
    train_loader, test_loader, original_point_clouds = data_generator(n_teeth=100)  

    # get model
    vae, _ = create_vae_net(latent_dim=16)
    
    # Test model for functionality
    X = next(iter(train_loader))[0] 
    X = X.view(X.size(0), -1)
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
        latent_dim=16
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
    reconstructed_tooth = reconstruct_data(vae, original_point_clouds, 5, False)
    
    # Generate new tooth data
    generated_tooth = generate_new_data(vae, False)


    # Calculate Chamfer Distance as evaluation metric
    original_points = original_point_clouds[0].numpy() # expected shape (1024, 3)
    reconstructed_points = reconstructed_tooth
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
    
    
    # Visualize latent space
    # ... 
    

if __name__ == "__main__":
    main()
