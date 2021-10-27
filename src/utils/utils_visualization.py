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

import matplotlib.pyplot as plt


def plot_reconstruction_ts(x, x_hat, n_features, filename):
    if n_features>1:
        fig, ax = plt.subplots(n_features, 1, figsize=(15, n_features))
        for i in range(n_features):
            ax[i].plot(x[i])
            ax[i].plot(x_hat[i])
            ax[i].grid()
    else:
        fig = plt.figure(figsize=(15,6))
        plt.plot(x[0])
        plt.plot(x_hat[0])
        plt.grid()
        
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close('all')

def plot_reconstruction_prob_ts(x, x_mu, x_sigma, n_features, filename):
    fig, ax = plt.subplots(n_features, 1, figsize=(15, n_features))
    for i in range(n_features):
        ax[i].plot(x[i])
        ax[i].plot(x_mu[i,:], color='black')
        ax[i].fill_between(range(len(x_mu[i,:])), x_mu[i,:] - x_sigma[i,:], x_mu[i,:] + x_sigma[i,:], color='blue', alpha=0.25)
        ax[i].grid()
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close('all')

def plot_anomaly_scores(score, labels, filename):
    fig, ax = plt.subplots(2, 1, figsize=(15,10))
    ax[0].plot(score, label='anomaly score')
    ax[1].plot(labels, label='label', c='orange')
    ax[0].grid()
    ax[1].grid()
    ax[0].legend()
    ax[1].legend()
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close('all')