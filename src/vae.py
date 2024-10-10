# -*- coding: utf-8 -*-
# @Author: ABr_hub
# @Date:   2024-10-03 22:33:19
# @Last Modified by:   ABr_hub
# @Last Modified time: 2024-10-10 15:26:35


import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def create_vae_net(
    latent_dim,
    vae_type,
):

    # Define the VAE architecture
    class VAE(nn.Module):
        
        if vae_type == 'linear':
            def __init__(self, latent_dim=latent_dim):
                super(VAE, self).__init__()
                self.latent_dim = latent_dim

                # Encoder layers
                self.encoder = nn.Sequential(
                    nn.Linear(1024 * 3, 512),
                    nn.ReLU(),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, latent_dim * 2)  # Output mean and log-variance
                )

                # Decoder layers
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, 512),
                    nn.ReLU(),
                    nn.Linear(512, 1024 * 3),
                    nn.Tanh()  # Output in range [-1, 1] for 3D coordinates
                )
        
        elif vae_type == 'cnn':
            def __init__(self, latent_dim=latent_dim):
                super(VAE, self).__init__()
                self.latent_dim = latent_dim
                
                # Encoder
                self.encoder = nn.Sequential(
                    nn.Conv3d(1, 32, kernel_size=4, stride=2, padding=1), # (B, 1, 32, 32, 32) -> (B, 32, 16, 16, 16)
                    nn.ReLU(),
                    nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1), # (B, 32, 16, 16, 16) -> (B, 64, 8, 8, 8)
                    nn.ReLU(),
                    nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1), # (B, 64, 8, 8, 8) -> (B, 128, 4, 4, 4)
                    nn.ReLU(),
                    nn.Flatten(),
                    nn.Linear(128 * 4 * 4 * 4, 256),
                    nn.ReLU(),
                    nn.Linear(256, latent_dim * 2) # Latent vector, output mean and log-variance
                )
                
                # Decoder (using Transpose Conv layers to upsample)
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, 256), 
                    nn.ReLU(),
                    nn.Linear(256, 128 * 4 * 4 * 4),
                    nn.ReLU(),
                    nn.Unflatten(dim=1, unflattened_size=(128, 4, 4, 4)),           # dim=1 to unflat (e.g. when tensor is (Batch, 2048) -> dim=1)
                                                                                    # unflattened_size = goal dimensions
                    nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
                    nn.ReLU(),
                    nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
                    nn.ReLU(),
                    nn.ConvTranspose3d(32, 1, kernel_size=4, stride=2, padding=1),
                    nn.Tanh(),  # Output in range [-1, 1] for 3D coordinates
                )


        def encode(self, x):
            '''
            The mean and logvar are the individual values for each of the latent
            dim. (I.e. each latent dim has its own mean and logvar).
            These are essentially learned with the neural network
            '''
            h = self.encoder(x)
            mean, logvar = torch.chunk(h, 2, dim=-1)
            return mean, logvar

        def decode(self, z):
            return self.decoder(z)

        def reparameterize(self, mean, logvar):
            '''
            Reparametrization trick.
            
            The mean and logvar are the outcome of the encoder part. This 
            is a vector for each, the mean and the logvar for each datapoint of 
            the latent dim. 
            
            The logvar is transformed into the standard deviation (std). 
            Furthermore random noise is added (eps; This has to be in the same
            shape as std).
            Altogether this allows the calculation of z which makes the 
            non-differentiable sampling from a gaussian distribution differentiable.
            
            Parameters
            ----------
            mean (torch.tensor) : Matrix of the shape (batch_size, latent_dim)
            logvar (torch.tensor) : Matrix of the shape (batch_size, latent_dim)
            
            Returns
            -------
            z (torch.tensor)  
            '''
            std = torch.exp(0.5 * logvar) 
            eps = torch.randn_like(std) # has to be of the same shape than std
            return mean + eps * std

        def forward(self, x):
            mean, logvar = self.encode(x)
            z = self.reparameterize(mean, logvar)
            recon_x = self.decode(z)
            return recon_x, mean, logvar

    # create a model instance
    net = VAE()
    
    # optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    
    return net, optimizer


 # Loss function (Reconstruction loss + KL divergence)
def vae_loss(
    recon_x, 
    x, 
    mean, 
    logvar
):
    '''
    The vea loss contains 2 main components:
    
        1. Reconstruction loss : Measures how well the decoder reconstructs
        the input data from the latent space
        
        2. KL-Divergence : Measures the difference between the distribution of 
        the latent variables and the normal distribution. (= Pressures the model
        to represent the latent variables as gaussian distribution).
    '''
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return recon_loss + kl_div


# Training loop
def train_vae(
    #model, 
    train_loader,
    dev_loader, 
    epochs=20,
    latent_dim=16,
    vae_type='linear'
):
    
    # initialize model and optimizer
    model, optimizer = create_vae_net(
        latent_dim=latent_dim,
        vae_type=vae_type
    )
    
    # send model to GPU
    model.to(device)
    
    # initialize losses 
    trainLoss = torch.zeros(epochs)
    devLoss =  torch.zeros(epochs)
    
    # loop over epochs
    for epoch in range(epochs):
        
        # set in training mode
        model.train()
        
        # initialize batch metrics
        batchAcc = []
        batchLoss = []
        
        # loop over training data batches
        for X in train_loader:
            
            # Usually the Dataloader returns a tuple (x,y); Since here are no y's
            # this step becomes necary
            X = X[0] 

            # push data to gpu
            X = X.to(device)
            #y = y.to(device)
            
            # x = batch[0].view(-1, 1024 * 3)  # Flatten the point cloud data

            # forward pass and loss
            recon_x, mean, logvar = model(X)
            loss = vae_loss(recon_x, X, mean, logvar)
            
            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # loss from this batch
            batchLoss.append(loss.item())
            
        # end of batch loop
        trainLoss[epoch] = np.mean(batchLoss)
        print(f'Epoch {epoch+1}, Loss: {trainLoss[epoch]}')


        ## test performance
        model.eval()
        
        X = next(iter(dev_loader))[0]
        
        # push to gpu
        X = X.to(device)
        #y = y.to(device)
        
        # forward pass and loss
        with torch.no_grad():
            recon_x, mean, logvar = model(X)
            loss = vae_loss(recon_x, X, mean, logvar)
        
        # and get average losses and accuracies across the batches
        devLoss[epoch] = loss.item()
            
    # end epochs
    return trainLoss, devLoss, model    


if __name__ == "__main__":
    # Generate dummy data 
    N = 1000    # N samples
    num_points = 32 # Datapoints per sample and x-y-z-coordinate
    vae_type='cnn'
    
    if vae_type == 'linear':
        dummy_pointcloud_tensors = torch.rand((N, num_points, 3))
    else:
        dummy_pointcloud_tensors = torch.rand((N, num_points, num_points, num_points))
    print('Dummy point clouds shape: ', dummy_pointcloud_tensors.shape)
    
    # Create a DataLoader for batching
    batch_size = 1
    dummy_loader = DataLoader(TensorDataset(dummy_pointcloud_tensors), batch_size=batch_size, shuffle=True)
    
    # Initialize the model, loss function, and optimizer
    vae, optimizer = create_vae_net(latent_dim=16, vae_type=vae_type)
    
    X = next(iter(dummy_loader))[0] # [0], to get only the input data
                                    # NOTE: Usually the dataloader gives back 
                                    # (x, labels) as tuple; Here only x values 
                                    # are fed into the Tensordataset
    
    # Flatten the tensor dimension since here only a simple mlp is defined
    if vae_type == 'linear':
        X = X.view(X.size(0), -1)
        
    # Reshape to cubic coordinates for the use in a 3d layer
    else: 
        X = X[:, :1000, :].view(X.size(0), 1, num_points, num_points, num_points)
                
    # Run dummy data thorugh model
    recon_x, mean, logvar = vae(X)
    print('\nValues: ', recon_x, mean, logvar)
    print('\nShapes: ', recon_x.shape, mean.shape, logvar.shape)
    
    loss = vae_loss(recon_x, X, mean, logvar)
    print('')
    print('Loss: ')
    print(loss)
    
    


    
