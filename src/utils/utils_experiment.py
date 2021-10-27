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

import os
import pickle
import numpy as np
import pandas as pd

import torch
from models.DGHL import DGHL

from utils.utils import de_unfold
from utils.utils_visualization import plot_reconstruction_ts, plot_anomaly_scores


def train_DGHL(mc, train_data, test_data, test_labels, train_mask, test_mask, entities, make_plots, root_dir):
    """
    train_data:
        List of tensors with training data, each shape (n_time, 1, n_features)
    test_data:
        List of tensor with training data, each shape (n_time, 1, n_features)
    test_labels:
        List of arrays with test lables, each (ntime)
    train_mask:
        List of tensors with training mask, each shape (n_time, 1, n_features)
    test_mask:
        List of tensors with test mask, each shape (n_time, 1, n_features)
    entities:
        List of names with entities
    """

    print(pd.Series(mc))
    # --------------------------------------- Random seed --------------------------------------
    np.random.seed(mc['random_seed'])

    # --------------------------------------- Parse paramaters --------------------------------------
    window_size = mc['window_size']
    window_hierarchy = mc['window_hierarchy']
    window_step = mc['window_step']
    n_features = mc['n_features']

    total_window_size = window_size*window_hierarchy

    # --------------------------------------- Data Processing --------------------------------------
    n_entities = len(entities)
    train_data_list = []
    test_data_list = []
    train_mask_list = []
    test_mask_list = []

    # Loop to pre-process each entity 
    for entity in range(n_entities):
        #print(10*'-','entity ', entity, ': ', entities[entity], 10*'-')
        train_data_entity = train_data[entity].copy()
        test_data_entity = test_data[entity].copy()
        train_mask_entity = train_mask[entity].copy()
        test_mask_entity = test_mask[entity].copy()

        assert train_data_entity.shape == train_mask_entity.shape, 'Train data and Train mask should have equal dimensions'
        assert test_data_entity.shape == test_mask_entity.shape, 'Test data and Test mask should have equal dimensions'
        assert train_data_entity.shape[2] == mc['n_features'], 'Train data should match n_features'
        assert test_data_entity.shape[2] == mc['n_features'], 'Test data should match n_features'

        # --------------------------------------- Data Processing ---------------------------------------
        # Complete first window for test, padding from training data
        padding = total_window_size - (len(test_data_entity) - total_window_size*(len(test_data_entity)//total_window_size))
        test_data_entity = np.vstack([train_data_entity[-padding:], test_data_entity])
        test_mask_entity = np.vstack([train_mask_entity[-padding:], test_mask_entity])

        # Create rolling windows
        train_data_entity = torch.Tensor(train_data_entity).float()
        train_data_entity = train_data_entity.permute(0,2,1)
        train_data_entity = train_data_entity.unfold(dimension=0, size=total_window_size, step=window_step)

        test_data_entity = torch.Tensor(test_data_entity).float()
        test_data_entity = test_data_entity.permute(0,2,1)
        test_data_entity = test_data_entity.unfold(dimension=0, size=total_window_size, step=window_step)

        train_mask_entity = torch.Tensor(train_mask_entity).float()
        train_mask_entity = train_mask_entity.permute(0,2,1)
        train_mask_entity = train_mask_entity.unfold(dimension=0, size=total_window_size, step=window_step)

        test_mask_entity = torch.Tensor(test_mask_entity).float()
        test_mask_entity = test_mask_entity.permute(0,2,1)
        test_mask_entity = test_mask_entity.unfold(dimension=0, size=total_window_size, step=window_step)

        train_data_list.append(train_data_entity)
        test_data_list.append(test_data_entity)
        train_mask_list.append(train_mask_entity)
        test_mask_list.append(test_mask_entity)

    # Append all windows for complete windows data
    train_windows_data = torch.vstack(train_data_list)
    train_windows_mask = torch.vstack(train_mask_list)

    # -------------------------------------------- Instantiate and train Model --------------------------------------------
    print('Training model...')
    model = DGHL(window_size=window_size, window_step=mc['window_step'], window_hierarchy=window_hierarchy,
                 hidden_multiplier=mc['hidden_multiplier'], max_filters=mc['max_filters'],
                 kernel_multiplier=mc['kernel_multiplier'], n_channels=n_features,
                 z_size=mc['z_size'], z_size_up=mc['z_size_up'], z_iters=mc['z_iters'],
                 z_sigma=mc['z_sigma'], z_step_size=mc['z_step_size'],
                 z_with_noise=mc['z_with_noise'], z_persistent=mc['z_persistent'],
                 batch_size=mc['batch_size'], learning_rate=mc['learning_rate'],
                 noise_std=mc['noise_std'],
                 normalize_windows=mc['normalize_windows'],
                 random_seed=mc['random_seed'], device=mc['device'])

    model.fit(X=train_windows_data, mask=train_windows_mask, n_iterations=mc['n_iterations'])
    
    # -------------------------------------------- Inference on each entity --------------------------------------------
    for entity in range(n_entities):

        rootdir_entity = f'{root_dir}/{entities[entity]}'
        os.makedirs(name=rootdir_entity, exist_ok=True)
        # Plots of reconstruction in train
        print('Reconstructing train...')
        x_train_true, x_train_hat, _, mask_windows = model.predict(X=train_data_list[entity], mask=train_mask_list[entity],
                                                                   z_iters=mc['z_iters_inference'])
        
        x_train_true, _ = de_unfold(x_windows=x_train_true, mask_windows=mask_windows, window_step=window_step)
        x_train_hat, _ = de_unfold(x_windows=x_train_hat, mask_windows=mask_windows, window_step=window_step)
        
        x_train_true = np.swapaxes(x_train_true,0,1)
        x_train_hat = np.swapaxes(x_train_hat,0,1)

        if make_plots:
            filename = f'{rootdir_entity}/reconstruction_train.png'
            plot_reconstruction_ts(x=x_train_true, x_hat=x_train_hat, n_features=n_features, filename=filename)

        # --------------------------------------- Inference on test and anomaly scores ---------------------------------------
        print('Computing scores on test...')
        score_windows, ts_score, x_windows, x_hat_windows, _, mask_windows = model.anomaly_score(X=test_data_list[entity],
                                                                                                 mask=test_mask_list[entity],
                                                                                                 z_iters=mc['z_iters_inference'])

        # Post-processing
        # Fold windows
        score_windows = score_windows[:,None,None,:]
        score_mask = np.ones(score_windows.shape)
        score, _ = de_unfold(x_windows=score_windows, mask_windows=score_mask, window_step=window_step)
        x_test_true, _ = de_unfold(x_windows=x_windows, mask_windows=mask_windows, window_step=window_step)
        x_test_hat, _ = de_unfold(x_windows=x_hat_windows, mask_windows=mask_windows, window_step=window_step)
        x_test_true = np.swapaxes(x_test_true,0,1)
        x_test_hat = np.swapaxes(x_test_hat,0,1)
        score = score.flatten()
        score = score[-len(test_labels[entity]):]

        if make_plots:
            filename = f'{rootdir_entity}/reconstruction_test.png'
            plot_reconstruction_ts(x=x_test_true, x_hat=x_test_hat, n_features=n_features, filename=filename)

        # Plot scores
        if make_plots:
            filename = f'{rootdir_entity}/anomaly_scores.png'
            plot_anomaly_scores(score=score, labels=test_labels[entity], filename=filename)

        results = {'score': score, 'ts_score':ts_score, 'x_test_true':x_test_true, 'x_test_hat':x_test_hat, 'labels':test_labels,
                    'x_train_true':x_train_true, 'x_train_hat':x_train_hat, 'train_mask': train_mask, 'mc':mc}

        with open(f'{rootdir_entity}/results.p','wb') as f:
            pickle.dump(results, f)