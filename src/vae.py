# -*- coding: utf-8 -*-
# @Author: Philipp N. Mueller
# @Date:   2024-10-03 22:33:19
# @Last Modified by:   Philipp N. Mueller
# @Last Modified time: 2024-10-05 10:41:33

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def create_vae_net(
    latent_dim
):

    # Define the VAE architecture
    class VAE(nn.Module):
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

        def encode(self, x):
            h = self.encoder(x)
            mean, logvar = torch.chunk(h, 2, dim=-1)
            return mean, logvar

        def decode(self, z):
            return self.decoder(z)

        def reparameterize(self, mean, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
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
):
    
    # initialize model and optimizer
    model, optimizer = create_vae_net(
        latent_dim=latent_dim
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
    # generate dummy data 
    dummy_point_clouds = np.random.random((20, 1024, 3))
    print('Dummy point clouds shape: ', dummy_point_clouds.shape)
    
    # convert dummy data to pytorch tensors ()
    dummy_pointcloud_tensors =  torch.tensor(dummy_point_clouds, dtype=torch.float32)
    
    # create a DataLoader for batching
    batch_size = 2
    dummy_loader = DataLoader(TensorDataset(dummy_pointcloud_tensors), batch_size=batch_size, shuffle=True)
    
    # Initialize the model, loss function, and optimizer
    vae, optimizer = create_vae_net(latent_dim=16)
    
    X = next(iter(dummy_loader))[0] # [0], to get only the input data
                                    # NOTE: Usually the dataloader gives back 
                                    # (x, labels) as tuple; Here only x values 
                                    # are fed into the Tensordataset
    
    # flatten the tensor dimension since here only a simple mlp is defined
    X = X.view(X.size(0), -1)
                
    # run dummy data thorugh model
    recon_x, mean, logvar = vae(X)
    print('\nValues: ', recon_x, mean, logvar)
    print('\nShapes: ', recon_x.shape, mean.shape, logvar.shape)
    
    loss = vae_loss(recon_x, X, mean, logvar)
    print('')
    print('Loss: ')
    print(loss)
    
    


    
