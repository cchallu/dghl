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

def get_random_occlusion_mask(dataset, n_intervals, occlusion_prob):
    len_dataset, _, n_features = dataset.shape

    interval_size = int(np.ceil(len_dataset/n_intervals))
    mask = np.ones(dataset.shape)
    for i in range(n_intervals):
        u = np.random.rand(n_features)
        mask_interval = (u>occlusion_prob)*1
        mask[i*interval_size:(i+1)*interval_size, :, :] = mask[i*interval_size:(i+1)*interval_size, :, :]*mask_interval

    # Add one random interval for complete missing features 
    feature_sum = mask.sum(axis=0)
    missing_features = np.where(feature_sum==0)[1]
    for feature in missing_features:
        i = np.random.randint(0, n_intervals)
        mask[i*interval_size:(i+1)*interval_size, :, feature] = 1

    return mask


def load_smd(entities, downsampling_size, occlusion_intervals, occlusion_prob, root_dir='./data', verbose=True):

    # ------------------------------------------------- Reading data -------------------------------------------------
    train_data = np.loadtxt(f'{root_dir}/ServerMachineDataset/train/{entities}.txt', delimiter=',')
    train_data = train_data[:, None, :]

    test_data = np.loadtxt(f'{root_dir}/ServerMachineDataset/test/{entities}.txt',delimiter=',')
    test_data = test_data[:, None, :]
    len_test = len(test_data)

    labels = np.loadtxt(f'{root_dir}/ServerMachineDataset/test_label/{entities}.txt',delimiter=',')

    dataset = np.vstack([train_data, test_data])

    if verbose:
        print('Full Train shape: ', train_data.shape)
        print('Full Test shape: ', test_data.shape)
        print('Full Dataset shape: ', dataset.shape)
        print('Full labels shape: ', labels.shape)
        print('---------')

    # ------------------------------------------------- Downsampling -------------------------------------------------
    # Padding
    right_padding = downsampling_size - dataset.shape[0]%downsampling_size
    dataset = np.pad(dataset, ((right_padding, 0), (0,0), (0,0) ))

    right_padding = downsampling_size - len_test%downsampling_size
    labels = np.pad(labels, (right_padding, 0))

    dataset = dataset.reshape(dataset.shape[0]//downsampling_size, -1, 1, dataset.shape[2]).max(axis=1)
    labels = labels.reshape(labels.shape[0]//downsampling_size, -1).max(axis=1)
    len_test_downsampled = int(np.ceil(len_test/downsampling_size))
    
    if verbose:
        print('Downsampled Dataset shape: ', dataset.shape)
        print('Downsampled labels: ', labels.shape)

    train_data = dataset[:-len_test_downsampled]
    test_data = dataset[-len_test_downsampled:]

    # ------------------------------------------------- Training Occlusion Mask -------------------------------------------------
    # Masks
    mask_filename = f'{root_dir}/ServerMachineDataset/train/mask_{entities}_{occlusion_intervals}_{occlusion_prob}.p'
    if os.path.exists(mask_filename):
        if verbose:
            print(f'Train mask {mask_filename} loaded!')
        train_mask = pickle.load(open(mask_filename,'rb'))
    else:
        print('Train mask not found, creating new one')
        train_mask = get_random_occlusion_mask(dataset=train_data, n_intervals=occlusion_intervals, occlusion_prob=occlusion_prob)
        with open(mask_filename,'wb') as f:
            pickle.dump(train_mask, f)
        if verbose:
            print(f'Train mask {mask_filename} created!')

    test_mask = np.ones(test_data.shape)

    if verbose:
        print('Train Data shape: ', train_data.shape)
        print('Test Data shape: ', test_data.shape)

        print('Train Mask mean: ', train_mask.mean())
        print('Test Mask mean: ', test_mask.mean())

    # Convert to lists
    train_data = [train_data]
    train_mask = [train_mask]
    test_data = [test_data]
    test_mask = [test_mask]
    labels = [labels]
    
    return train_data, train_mask, test_data, test_mask, labels


def load_nasa(entities, downsampling_size, occlusion_intervals, occlusion_prob, root_dir='./data', verbose=True):
    
    train_data_list = []
    test_data_list = []
    test_labels_list = []
    train_mask_list = []
    test_mask_list = []

    for entity in entities:
        # ------------------------------------------------- Reading data -------------------------------------------------
        if verbose:
            print(10*'-','entity ', entity, 10*'-')
        train_data = np.load(f'{root_dir}/NASA/train/{entity}.npy')
        test_data = np.load(f'{root_dir}/NASA/test/{entity}.npy')
        labels = np.load(f'{root_dir}/NASA/labels/{entity}.npy')
        train_data = train_data[:, None, :]
        test_data = test_data[:, None, :]
        len_test = len(test_data)

        dataset = np.vstack([train_data, test_data])
        if verbose:
            print('Full Train shape: ', train_data.shape)
            print('Full Test shape: ', test_data.shape)
            print('Full Dataset shape: ', dataset.shape)
            print('Full labels shape: ', labels.shape)
            print('---------')

        # ------------------------------------------------- Downsampling -------------------------------------------------
        # Padding
        if downsampling_size > 1:
            right_padding = downsampling_size - dataset.shape[0]%downsampling_size
        else:
            right_padding = 0
        dataset = np.pad(dataset, ((right_padding, 0), (0,0), (0,0) ))

        if downsampling_size > 1:
            right_padding = downsampling_size - len_test%downsampling_size
        else:
            right_padding = 0
        labels = np.pad(labels, (right_padding, 0))

        dataset = dataset.reshape(dataset.shape[0]//downsampling_size, -1, 1, dataset.shape[2]).max(axis=1)
        labels = labels.reshape(labels.shape[0]//downsampling_size, -1).max(axis=1)
        len_test_downsampled = int(np.ceil(len_test/downsampling_size))

        if verbose:
            print('Downsampled Dataset shape: ', dataset.shape)
            print('Downsampled labels: ', labels.shape)

        train_data = dataset[:-len_test_downsampled]
        test_data = dataset[-len_test_downsampled:]

        # ------------------------------------------------- Training Occlusion Mask -------------------------------------------------
        # Masks
        mask_filename = f'{root_dir}/NASA/train/mask_{entity}_{occlusion_intervals}_{occlusion_prob}.p'
        if os.path.exists(mask_filename):
            if verbose:
                print(f'Train mask {mask_filename} loaded!')
            train_mask = pickle.load(open(mask_filename,'rb'))
        else:
            print('Train mask not found, creating new one')
            train_mask = get_random_occlusion_mask(dataset=train_data, n_intervals=occlusion_intervals, occlusion_prob=occlusion_prob)
            with open(mask_filename,'wb') as f:
                pickle.dump(train_mask, f)
            if verbose:
                print(f'Train mask {mask_filename} created!')

        test_mask = np.ones(test_data.shape)

        if verbose:
            print('Train Data shape: ', train_data.shape)
            print('Test Data shape: ', test_data.shape)

            print('Train Mask mean: ', train_mask.mean())
            print('Test Mask mean: ', test_mask.mean())

        train_data_list.append(train_data)
        test_data_list.append(test_data)
        test_labels_list.append(labels)
        train_mask_list.append(train_mask)
        test_mask_list.append(test_mask)
    
    return train_data_list, train_mask_list, test_data_list, test_mask_list, test_labels_list
    

def load_swat(downsampling_size, occlusion_intervals, occlusion_prob, root_dir='./data', verbose=True):

    train_data = pd.read_csv(f'{root_dir}/SWAT/swat_train.csv').values
    train_data = train_data[:, None, :]
    test_data = pd.read_csv(f'{root_dir}/SWAT/swat_test.csv').values
    test_data = test_data[:, None, :]
    len_test = len(test_data)
    test_labels = pd.read_csv(f'{root_dir}/SWAT/labels_test.csv').values

    dataset = np.vstack([train_data, test_data])
    if verbose:
        print('Full Train shape: ', train_data.shape)
        print('Full Test shape: ', test_data.shape)
        print('Full Dataset shape: ', dataset.shape)
        print('Full labels shape: ', test_labels.shape)
        print('---------')

    # Padding
    right_padding = downsampling_size - dataset.shape[0]%downsampling_size
    dataset = np.pad(dataset, ((right_padding, 0), (0,0), (0,0) ))

    right_padding = downsampling_size - len_test%downsampling_size
    test_labels = np.pad(test_labels, (right_padding, 0))

    dataset = dataset.reshape(dataset.shape[0]//downsampling_size, -1, 1, dataset.shape[2]).max(axis=1)
    test_labels = test_labels.reshape(test_labels.shape[0]//downsampling_size, -1).max(axis=1)
    len_test_downsampled = int(np.ceil(len_test/downsampling_size))
    
    if verbose:
        print('Downsampled Dataset shape: ', dataset.shape)
        print('Downsampled labels: ', test_labels.shape)

    train_data = dataset[:-len_test_downsampled]
    test_data = dataset[-len_test_downsampled:]

    train_max = train_data.max(axis=0, keepdims=True)

    train_data = train_data/train_max
    test_data = test_data/train_max

    # ------------------------------------------------- Training Occlusion Mask -------------------------------------------------
    # Masks
    mask_filename = f'{root_dir}/SWAT/mask_{occlusion_intervals}_{occlusion_prob}.p'
    if os.path.exists(mask_filename):
        if verbose:
            print(f'Train mask {mask_filename} loaded!')
        train_mask = pickle.load(open(mask_filename,'rb'))
    else:
        print('Train mask not found, creating new one')
        train_mask = get_random_occlusion_mask(dataset=train_data, n_intervals=occlusion_intervals, occlusion_prob=occlusion_prob)
        with open(mask_filename,'wb') as f:
            pickle.dump(train_mask, f)
        if verbose:
            print(f'Train mask {mask_filename} created!')

    test_mask = np.ones(test_data.shape)

    if verbose:
            print('Train Data shape: ', train_data.shape)
            print('Test Data shape: ', test_data.shape)

            print('Train Mask mean: ', train_mask.mean())
            print('Test Mask mean: ', test_mask.mean())

    train_data = [train_data]
    train_mask = [train_mask]
    test_data = [test_data]
    test_mask = [test_mask]
    test_labels = [test_labels]

    return train_data, train_mask, test_data, test_mask, test_labels


