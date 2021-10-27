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

class DGHL(object):
    def __init__(self, window_size, window_step, window_hierarchy,
                 n_channels, hidden_multiplier, max_filters, kernel_multiplier,
                 z_size, z_size_up, z_iters, z_sigma, z_step_size, z_with_noise, z_persistent,
                 batch_size, learning_rate, noise_std, normalize_windows,
                 random_seed, device=None):
        super(DGHL, self).__init__()

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

        # Alternating back-propagation
        self.z_iters = z_iters
        self.z_sigma = z_sigma
        self.z_step_size = z_step_size
        self.z_with_noise = z_with_noise
        self.z_persistent = z_persistent

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

    def get_z(self, z, x, m, n_iters, with_noise):
        
        mse = nn.MSELoss(reduction='sum')
        z_u = z[0]
        z_l = z[1]

        for i in range(n_iters):
            z_u = torch.autograd.Variable(z_u, requires_grad=True)
            z_l = torch.autograd.Variable(z_l, requires_grad=True)
            
            z_u_repeated = torch.repeat_interleave(z_u, self.window_hierarchy, 0)
            z = torch.cat((z_u_repeated, z_l), dim=1).to(self.device)

            x_hat = self.generator(z, m)

            L = 1.0 / (2.0 * self.z_sigma * self.z_sigma) * mse(x_hat, x)
            L.backward()
            z_u = z_u - 0.5 * self.z_step_size * self.z_step_size * (z_u + z_u.grad)
            z_l = z_l - 0.5 * self.z_step_size * self.z_step_size * (z_l + z_l.grad)
            if with_noise:
                eps_u = torch.randn(len(z_u), self.z_size_up, 1, 1).to(z_u.device)
                z_u += self.z_step_size * eps_u
                eps_l = torch.randn(len(z_l), self.z_size, 1, 1).to(z_l.device)
                z_l += self.z_step_size * eps_l

        z_u = z_u.detach()
        z_l = z_l.detach()
        z = z.detach()

        return z, z_u, z_l

    def sample_gaussian(self, n_dim, n_samples):
        p_0 = torch.distributions.MultivariateNormal(torch.zeros(n_dim), 0.01*torch.eye(n_dim))
        p_0 = p_0.sample([n_samples]).view([n_samples, -1, 1, 1])

        return p_0

    def get_batch(self, X, mask, batch_size, p_0_chains_u, p_0_chains_l, z_persistent, shuffle=True):
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
        
        if z_persistent:
            p_0_z_u = p_0_chains_u[i]
            p_0_z_l = p_0_chains_l[i]
            p_0_z_u = p_0_z_u.reshape(batch_size, self.z_size_up,1,1)
            p_0_z_l = p_0_z_l.reshape(batch_size*self.window_hierarchy, self.z_size,1,1)
        else:
            p_0_z_u = self.sample_gaussian(n_dim=self.z_size_up, n_samples=batch_size)
            p_0_z_u = p_0_z_u.to(self.device)

            p_0_z_l = self.sample_gaussian(n_dim=self.z_size, n_samples=batch_size*self.window_hierarchy)
            p_0_z_l = p_0_z_l.to(self.device)

        p_d_x = torch.Tensor(p_d_x).to(self.device)
        p_d_m = torch.Tensor(p_d_m).to(self.device)
        x_scales = x_scales.to(self.device)

        p_0_z = [p_0_z_u, p_0_z_l]

        return p_d_x, p_0_z, p_d_m, i, x_scales

    def fit(self, X, mask, n_iterations):
        if self.z_persistent:
            self.p_0_chains_u = torch.zeros((X.shape[0],1,self.z_size_up,1,1))
            self.p_0_chains_l = torch.zeros((X.shape[0], self.window_hierarchy, self.z_size,1,1))
            for i in range(X.shape[0]):
                p_0_chains_u = self.sample_gaussian(n_dim=self.z_size_up, n_samples=1)
                p_0_chains_u = p_0_chains_u.to(self.device)
                self.p_0_chains_u[i] = p_0_chains_u

                p_0_chains_l = self.sample_gaussian(n_dim=self.z_size, n_samples=self.window_hierarchy)
                p_0_chains_l = p_0_chains_l.to(self.device)
                self.p_0_chains_l[i] = p_0_chains_l
            
        else:
            self.p_0_chains_u = None
            self.p_0_chains_l = None

        optim = torch.optim.Adam(self.generator.parameters(), lr=self.learning_rate, betas=[.9, .999])
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=int(np.ceil(n_iterations/3)), gamma=0.8)

        mse = nn.MSELoss(reduction='sum')
        # Training loop
        mse_list = []
        start_time = time.time()
        for i in range(n_iterations):
            self.generator.train()
            # Sample windows
            x, z_0, m, chains_i, x_scales = self.get_batch(X=X, mask=mask, batch_size=self.batch_size, p_0_chains_u=self.p_0_chains_u,
                                                 p_0_chains_l=self.p_0_chains_l,
                                                 z_persistent=self.z_persistent, shuffle=True)
            x = x + self.noise_std*(torch.randn(x.shape).to(self.device))

            # Sample z with Langevin Dynamics
            z, z_u, z_l = self.get_z(z=z_0, x=x, m=m, n_iters=self.z_iters, with_noise=self.z_with_noise)
            x_hat = self.generator(z, m)

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
            L = 0.5 * self.z_sigma * self.z_sigma * mse(x, x_hat)
            optim.zero_grad()
            L.backward()
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 5.0)
            optim.step()
            lr_scheduler.step()
            mse_list.append(L.cpu().data.numpy())

            if self.z_persistent:
                z_u = z_u.reshape(self.batch_size,1,self.z_size_up,1,1)
                z_l = z_l.reshape(self.batch_size,self.window_hierarchy,self.z_size,1,1)
                self.p_0_chains_u[chains_i] = z_u
                self.p_0_chains_l[chains_i] = z_l

            if i % 50 == 0:
                norm_z0 = torch.norm(z_0[0], dim=0).mean()
                norm_z = torch.norm(z, dim=0).mean()
                batch_size = len(x)
                print('{:>6d} mse(x, x_hat)={:>10.4f} norm(z0)={:>10.4f} norm(z)={:>10.4f} time={:>10.4f}'.format(i, np.mean(mse_list) / batch_size,
                                                                                                                  norm_z0, norm_z,
                                                                                                                  time.time()-start_time))
                mse_list = []

        self.generator.eval()
    
    def predict(self, X, mask, z_iters):
        self.generator.eval()

        # Get full batch
        x, z_0, m, _, x_scales = self.get_batch(X=X, mask=mask, batch_size=len(X), p_0_chains_u=None, p_0_chains_l=None, z_persistent=False, shuffle=False)

        # Forward
        z, _, _ = self.get_z(z=z_0, x=x, m=m, n_iters=z_iters, with_noise=False)
        m = torch.ones(m.shape).to(self.device) # In forward of generator, mask is all ones to reconstruct everything
        x_hat = self.generator(z, m)

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
        z = z.cpu().data.numpy()
        m = m.cpu().data.numpy()

        return x, x_hat, z, m

    def anomaly_score(self, X, mask, z_iters):
        x, x_hat, z, mask = self.predict(X=X, mask=mask, z_iters=z_iters)
        x_hat = x_hat*mask # Hide non-available data

        x_flatten = x.squeeze(2)
        x_hat_flatten = x_hat.squeeze(2)
        mask_flatten = mask.squeeze(2)
        z = z.squeeze((2,3))

        ts_score = np.square(x_flatten-x_hat_flatten)

        score = np.average(ts_score, axis=1, weights=mask_flatten)

        return score, ts_score, x, x_hat, z, mask