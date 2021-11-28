"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
  
  Licensed under the Apache License, Version 2.0 (the "License").
  You may not use this file except in compliance with the License.
  You may obtain a copy of the License at
  
      http://www.apache.org/licenses/LICENSE-2.0
  
  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
"""

import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torchvision as tv
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, window_size=32, hidden_multiplier=32, latent_size=100, n_channels=3, max_filters=256, kernel_multiplier=1):
        super(Generator, self).__init__()

        n_layers = int(np.log2(window_size))
        layers = []
        filters_list = []
        # First layer
        filters = min(max_filters, hidden_multiplier*(2**(n_layers-2)))
        layers.append(nn.ConvTranspose1d(in_channels=latent_size, out_channels=filters,
                                         kernel_size=4, stride=1, padding=0, bias=False))
        layers.append(nn.BatchNorm1d(filters))
        filters_list.append(filters)
        # Hidden layers
        for i in reversed(range(1, n_layers-1)):
            filters = min(max_filters, hidden_multiplier*(2**(i-1)))
            layers.append(nn.ConvTranspose1d(in_channels=filters_list[-1], out_channels=filters,
                                             kernel_size=4*kernel_multiplier, stride=2, padding=1 + (kernel_multiplier-1)*2, bias=False))
            layers.append(nn.BatchNorm1d(filters))
            layers.append(nn.ReLU())
            filters_list.append(filters)

        # Output layer
        layers.append(nn.ConvTranspose1d(in_channels=filters_list[-1], out_channels=n_channels, kernel_size=3, stride=1, padding=1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x, m=None):
        x = x[:,:,0,:]
        x = self.layers(x)
        x = x[:,:,None,:]
        
        # Hide mask
        if m is not None:
            x = x * m

        return x

class Encoder(nn.Module):
    def __init__(self, window_size=64*4, hidden_multiplier=32, latent_size=100, n_channels=3, max_filters=256):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=n_channels, out_channels=8, kernel_size=3, stride=2, padding=1)
        self.batch1 = nn.BatchNorm1d(8)

        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.batch2 = nn.BatchNorm1d(16)
        self.max_pooling1 = nn.MaxPool1d(3, stride=2, padding=1)

        self.conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.batch3 = nn.BatchNorm1d(32)
        self.max_pooling2 = nn.MaxPool1d(3, stride=2, padding=1)

        self.linear1 = nn.Linear(256, 128)
        self.linear2 = nn.Linear(128, latent_size)
        self.linear3 = nn.Linear(128, latent_size)

    def forward(self, x, m=None):
        x = x[:,:,0,:]

        x = self.conv1(x)
        x = self.batch1(x)
        #print('x.shape', x.shape)

        x = self.conv2(x)
        x = self.batch2(x)
        x = F.relu(x)
        #print('x.shape', x.shape)

        x = self.max_pooling1(x)
        #print('x.shape', x.shape)

        x = self.conv3(x)
        x = self.batch3(x)
        x = F.relu(x)
        #print('x.shape', x.shape)

        x = self.max_pooling2(x)
        #print('x.shape', x.shape)

        x = torch.flatten(x, start_dim=1)
        #print('x.shape', x.shape)

        x = self.linear1(x)

        mu = self.linear2(x)
        log_var = self.linear3(x)

        #x = x[:,:,None,:]

        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu #+ (eps * std) # sampling as if coming from the input space

        sample = sample[:, :, None, None] # Adds 2 dim
        
        #print('sample.shape', sample.shape)
        
        return sample, std


class DGHL_encoder(object):
    def __init__(self, window_size, window_step, window_hierarchy,
                 n_channels, hidden_multiplier, max_filters, kernel_multiplier,
                 z_size, z_size_up,
                 batch_size, learning_rate, noise_std, normalize_windows,
                 random_seed, device=None):
        super(DGHL_encoder, self).__init__()

        # Generator
        self.window_size = window_size # Not used for now
        self.window_step = window_step # Not used for now
        self.n_channels = n_channels
        self.hidden_multiplier = hidden_multiplier
        self.z_size = z_size
        self.z_size_up = z_size_up
        self.max_filters = max_filters
        self.kernel_multiplier = kernel_multiplier
        self.normalize_windows = normalize_windows

        self.window_hierarchy = window_hierarchy

        # Training
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.noise_std = noise_std

        # Generator
        torch.manual_seed(random_seed)

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.generator = Generator(window_size=self.window_size, hidden_multiplier=self.hidden_multiplier,
                                   latent_size=self.z_size+self.z_size_up,
                                   n_channels=self.n_channels, max_filters=self.max_filters,
                                   kernel_multiplier=self.kernel_multiplier).to(self.device)

        self.encoder = Encoder(window_size=window_size, hidden_multiplier=hidden_multiplier,
                               latent_size=self.z_size+self.z_size_up, n_channels=n_channels,
                               max_filters=max_filters).to(self.device)


    def get_batch(self, X, mask, batch_size, shuffle=True):
        """
        X tensor of shape (n_windows, n_features, 1, window_size*A_L)
        """
        if shuffle:
            i = torch.LongTensor(batch_size).random_(0, X.shape[0])
        else:
            i = torch.LongTensor(range(batch_size))

        p_d_x = X[i]
        p_d_m = mask[i]

        x_scales = p_d_x[:,:,:,[0]]
        if self.normalize_windows:
            p_d_x = p_d_x - x_scales

        # Wrangling from (batch_size, n_features, 1, window_size*window_hierarchy) -> (batch_size*window_hierarchy, n_features, 1, window_size)
        p_d_x = p_d_x.unfold(dimension=-1, size=self.window_size, step=self.window_size)
        p_d_x = p_d_x.swapaxes(1,3)
        p_d_x = p_d_x.swapaxes(2,3)
        p_d_x = p_d_x.reshape(batch_size*self.window_hierarchy, self.n_channels, 1, self.window_size)

        p_d_m = p_d_m.unfold(dimension=-1, size=self.window_size, step=self.window_size)
        p_d_m = p_d_m.swapaxes(1,3)
        p_d_m = p_d_m.swapaxes(2,3)
        p_d_m = p_d_m.reshape(batch_size*self.window_hierarchy, self.n_channels, 1, self.window_size)

        # Hide with mask
        p_d_x = p_d_x * p_d_m

        p_d_x = torch.Tensor(p_d_x).to(self.device)
        p_d_m = torch.Tensor(p_d_m).to(self.device)
        x_scales = x_scales.to(self.device)

        return p_d_x, p_d_m, x_scales

    def fit(self, X, mask, n_iterations):

        optim = torch.optim.Adam(self.generator.parameters(), lr=self.learning_rate, betas=[.9, .999])
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=int(np.ceil(n_iterations/3)), gamma=0.8)

        mse = nn.MSELoss(reduction='sum')
        # Training loop
        mse_list = []
        start_time = time.time()
        for i in range(n_iterations):
            self.generator.train()
            # Sample windows
            x, m, x_scales = self.get_batch(X=X, mask=mask, batch_size=self.batch_size, shuffle=True)
            x = x + self.noise_std*(torch.randn(x.shape).to(self.device))

            # Sample z with Langevin Dynamics
            #z, z_u, z_l = self.get_z(z=z_0, x=x, m=m, n_iters=self.z_iters, with_noise=self.z_with_noise)
            mu, logvar = self.encoder(x)

            x_hat = self.generator(mu, m)

            # Return to window_size * window_hierarchy size
            x = x.swapaxes(0,2)
            x = x.reshape(1,self.n_channels,-1, self.window_size*self.window_hierarchy)
            x = x.swapaxes(0,2)

            x_hat = x_hat.swapaxes(0,2)
            x_hat = x_hat.reshape(1,self.n_channels,-1, self.window_size*self.window_hierarchy)
            x_hat = x_hat.swapaxes(0,2)

            m = m.swapaxes(0,2)
            m = m.reshape(1,self.n_channels,-1, self.window_size*self.window_hierarchy)
            m = m.swapaxes(0,2)

            if self.normalize_windows:
                x = x + x_scales
                x_hat = x_hat + x_scales
                x = x * m
                x_hat = x_hat * m

            # Loss and update
            L = 0.5 * mse(x, x_hat)
            optim.zero_grad()
            L.backward()
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 5.0)
            optim.step()
            lr_scheduler.step()
            mse_list.append(L.cpu().data.numpy())

            if i % 50 == 0:
                batch_size = len(x)
                print('{:>6d} mse(x, x_hat)={:>10.4f} time={:>10.4f}'.format(i, np.mean(mse_list) / batch_size, time.time()-start_time))
                mse_list = []

        self.generator.eval()
    
    def predict(self, X, mask):
        self.generator.eval()

        # Get full batch
        x, m, x_scales = self.get_batch(X=X, mask=mask, batch_size=len(X), shuffle=False)

        # Forward
        #z, _, _ = self.get_z(z=z_0, x=x, m=m, n_iters=z_iters, with_noise=False)
        mu, logvar = self.encoder(x)
        m = torch.ones(m.shape).to(self.device) # In forward of generator, mask is all ones to reconstruct everything
        x_hat = self.generator(mu, m)

        # Return to window_size * window_hierarchy size
        x = x.swapaxes(0,2)
        x = x.reshape(1,self.n_channels,-1, self.window_size*self.window_hierarchy)
        x = x.swapaxes(0,2)

        x_hat = x_hat.swapaxes(0,2)
        x_hat = x_hat.reshape(1,self.n_channels,-1, self.window_size*self.window_hierarchy)
        x_hat = x_hat.swapaxes(0,2)

        m = m.swapaxes(0,2)
        m = m.reshape(1,self.n_channels,-1, self.window_size*self.window_hierarchy)
        m = m.swapaxes(0,2)

        x = x.cpu().data.numpy()
        x_hat = x_hat.cpu().data.numpy()
        mu = mu.cpu().data.numpy()
        m = m.cpu().data.numpy()

        return x, x_hat, mu, m

    def anomaly_score(self, X, mask):
        x, x_hat, mu, mask = self.predict(X=X, mask=mask)
        x_hat = x_hat*mask # Hide non-available data

        x_flatten = x.squeeze(2)
        x_hat_flatten = x_hat.squeeze(2)
        mask_flatten = mask.squeeze(2)
        mu = mu.squeeze((2,3))

        ts_score = np.square(x_flatten-x_hat_flatten)

        score = np.average(ts_score, axis=1, weights=mask_flatten)

        return score, ts_score, x, x_hat, mu, mask