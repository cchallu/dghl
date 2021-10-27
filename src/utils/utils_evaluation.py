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

import pickle
import glob

import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors

from utils.utils import f1_score, best_f1_linspace, normalize_scores

MACHINES = ['machine-1-1','machine-1-2','machine-1-3','machine-1-4','machine-1-5','machine-1-6','machine-1-7','machine-1-8',
            'machine-2-1', 'machine-2-2','machine-2-3','machine-2-4','machine-2-5','machine-2-6','machine-2-7','machine-2-8','machine-2-9', 
            'machine-3-1', 'machine-3-2', 'machine-3-3', 'machine-3-4','machine-3-5','machine-3-6','machine-3-7','machine-3-8', 'machine-3-9','machine-3-10', 'machine-3-11']

# ------------------------------------------------------- SMD -------------------------------------------------------
def smd_load_scores(scores_dir, data_dir, machines):
    labels_list = []
    scores_list = []
    for machine in machines:
        results_file = f'{scores_dir}/SMD/{machine}/results.p'
        results = pickle.load(open(results_file,'rb'))

        scores = results['score']
        labels = np.loadtxt(f'{data_dir}/ServerMachineDataset/test_label/{machine}.txt',delimiter=',')
        scores = scores.repeat(10)[-len(labels):]
        
        assert scores.shape == labels.shape, 'Wrong dimensions'

        labels_list.append(labels)
        scores_list.append(scores)
        
    return scores_list, labels_list 

def smd_compute_f1(scores_dir, data_dir):

    scores, labels = smd_load_scores(scores_dir=scores_dir, data_dir=data_dir, machines=MACHINES)

    scores_normalized = normalize_scores(scores=scores, interval_size=64*4*10) # 64*1*10

    scores_normalized = np.hstack(scores_normalized)
    labels = np.hstack(labels)

    f1, precision, recall, *_ = best_f1_linspace(scores=scores_normalized, labels=labels, n_splits=1000, segment_adjust=True) #1000

    return f1, precision, recall

def smd_one_liner(data_dir):
    labels_list = []
    scores_list = []
    for machine in MACHINES:
        train_data = np.loadtxt(f'{data_dir}/ServerMachineDataset/train/{machine}.txt',delimiter=',')
        test_data = np.loadtxt(f'{data_dir}/ServerMachineDataset/test/{machine}.txt',delimiter=',')

        train_means = train_data.mean(axis=0, keepdims=True)
        scores = np.abs(test_data - train_means)
        scores = scores.mean(axis=1)

        labels = np.loadtxt(f'{data_dir}/ServerMachineDataset/test_label/{machine}.txt',delimiter=',')
        labels_list.append(labels)
        scores_list.append(scores)

    scores_normalized = normalize_scores(scores=scores_list, interval_size=64*4*10) # 64*1*10

    scores_normalized = np.hstack(scores_normalized)
    labels = np.hstack(labels_list)

    f1, precision, recall, *_ = best_f1_linspace(scores=scores_normalized, labels=labels, n_splits=100, segment_adjust=True)
        
    return f1, precision, recall

def smd_nn(data_dir):
    downsampling_size = 10
    labels_list = []
    scores_list = []
    for machine in MACHINES:
        train_data = np.loadtxt(f'{data_dir}/ServerMachineDataset/train/{machine}.txt',delimiter=',')
        train_data = train_data[:, None, :]
        test_data = np.loadtxt(f'{data_dir}/ServerMachineDataset/test/{machine}.txt',delimiter=',')
        test_data = test_data[:, None, :]
        len_test = len(test_data)

        # Padding
        dataset = np.vstack([train_data, test_data])
        
        right_padding = downsampling_size - dataset.shape[0]%downsampling_size
        dataset = np.pad(dataset, ((right_padding, 0), (0,0), (0,0) ))

        right_padding = downsampling_size - len_test%downsampling_size

        dataset = dataset.reshape(dataset.shape[0]//downsampling_size, -1, 1, dataset.shape[2]).max(axis=1)
        len_test_downsampled = int(np.ceil(len_test/downsampling_size))

        train_data = dataset[:-len_test_downsampled, 0, :]
        test_data = dataset[-len_test_downsampled:, 0, :]

        nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(train_data)
        distances, _ = nbrs.kneighbors(test_data)
        scores = distances.mean(axis=1)        

        labels = np.loadtxt(f'{data_dir}/ServerMachineDataset/test_label/{machine}.txt',delimiter=',')
        scores = scores.repeat(10)[-len(labels):]

        labels_list.append(labels)
        scores_list.append(scores)

    scores_normalized = normalize_scores(scores=scores_list, interval_size=64*4*10)
    scores_normalized = np.hstack(scores_normalized)
    labels = np.hstack(labels_list)

    f1, precision, recall, *_ = best_f1_linspace(scores=scores_normalized, labels=labels, n_splits=100, segment_adjust=True)
        
    return f1, precision, recall

# ------------------------------------------------------- NASA -------------------------------------------------------
def nasa_load_scores(dataset, scores_dir, data_dir):
    meta_data = pd.read_csv(f'{data_dir}/NASA/labeled_anomalies.csv')
    entities = meta_data[meta_data['spacecraft']==dataset]['chan_id'].values

    labels_list = []
    scores_list = []
    for entity in entities:
        results_file = f'{scores_dir}/{dataset}/{entity}/results.p'
        results = pickle.load(open(results_file,'rb'))

        ts_scores = results['ts_score']
        labels = np.load(f'{data_dir}/NASA/labels/{entity}.npy')
        ts_scores = np.swapaxes(ts_scores,0,1)
        ts_scores = ts_scores.reshape(len(ts_scores),-1)
        ts_scores = ts_scores[:, -len(labels):]
        score = ts_scores[0]
        
        assert score.shape == labels.shape, 'Wrong dimensions'
        
        labels_list.append(labels)
        scores_list.append(score)

    return scores_list, labels_list

def nasa_compute_f1(dataset, scores_dir, data_dir):

    scores, labels = nasa_load_scores(dataset=dataset, scores_dir=scores_dir, data_dir=data_dir)

    if dataset=='SMAP':
        scores_normalized = normalize_scores(scores=scores, interval_size=64*4)
    else:
        scores_normalized = normalize_scores(scores=scores, interval_size=10000)

    labels = np.hstack(labels)
    scores = np.hstack(scores)
    scores_normalized = np.hstack(scores_normalized)

    scores_max = scores_normalized.max()
    if np.isinf(scores_max) or np.isnan(scores_max):
        f1 = None
        precision = None
        recall = None
    else:
        f1, precision, recall, *_ = best_f1_linspace(scores=scores_normalized, labels=labels, n_splits=1000, segment_adjust=True)

    return f1, precision, recall

def nasa_one_liner(dataset, data_dir):
    meta_data = pd.read_csv(f'{data_dir}/NASA/labeled_anomalies.csv')
    entities = meta_data[meta_data['spacecraft']==dataset]['chan_id'].values

    labels_list = []
    scores_list = []
    for entity in entities:
        train_data = np.load(f'{data_dir}/NASA/train/{entity}.npy')        
        test_data = np.load(f'{data_dir}/NASA/test/{entity}.npy')
        labels = np.load(f'{data_dir}/NASA/labels/{entity}.npy')

        train_means = train_data.mean(axis=0, keepdims=True)
        scores = np.abs(test_data - train_means)[:,0]

        labels_list.append(labels)
        scores_list.append(scores)

    scores_normalized = normalize_scores(scores=scores_list, interval_size=10000)

    scores_normalized = np.hstack(scores_normalized)
    labels = np.hstack(labels_list)

    f1, precision, recall, *_ = best_f1_linspace(scores=scores_normalized, labels=labels, n_splits=100, segment_adjust=True)
        
    return f1, precision, recall

def nasa_nn(dataset, data_dir):
    meta_data = pd.read_csv(f'{data_dir}/NASA/labeled_anomalies.csv')
    entities = meta_data[meta_data['spacecraft']==dataset]['chan_id'].values

    labels_list = []
    scores_list = []
    for entity in entities:
        train_data = np.load(f'{data_dir}/NASA/train/{entity}.npy')        
        test_data = np.load(f'{data_dir}/NASA/test/{entity}.npy')
        labels = np.load(f'{data_dir}/NASA/labels/{entity}.npy')

        nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(train_data)
        distances, _ = nbrs.kneighbors(test_data)
        scores = distances.mean(axis=1)

        labels_list.append(labels)
        scores_list.append(scores)

    scores_normalized = normalize_scores(scores=scores_list, interval_size=10000)

    scores_normalized = np.hstack(scores_normalized)
    labels = np.hstack(labels_list)

    f1, precision, recall, *_ = best_f1_linspace(scores=scores_normalized, labels=labels, n_splits=100, segment_adjust=True)
        
    return f1, precision, recall

# ------------------------------------------------------- SWAT -------------------------------------------------------
def swat_compute_f1(results_dir):

    results_file = f'{results_dir}/SWAT/results.p'
    results = pickle.load(open(results_file,'rb'))

    labels = pd.read_csv(f'../data/SWAT/labels_test.csv').values

    ts_scores = results['ts_score']
    len_test = len(labels)//2
    labels = labels[-len_test:, 0]

    x_train_true = results['x_train_true']
    x_train_hat = results['x_train_hat']
    train_errors = np.square(x_train_hat-x_train_true)
    train_errors = train_errors.mean(axis=1)
    train_errors = 100*train_errors[:,None]

    ts_scores = np.swapaxes(ts_scores,0,1)
    ts_scores = ts_scores.reshape(len(ts_scores),-1)
    ts_scores = ts_scores.repeat(10,axis=1)
    ts_scores = ts_scores[:, -len(labels):]

    scores_new = ts_scores/(train_errors)
    scores_new[10,:] = 0
    scores_new = scores_new.mean(axis=0)

    f1, precision, recall, *_ = best_f1_linspace(scores=scores_new, labels=labels, n_splits=1000, segment_adjust=False)

    return f1, precision, recall

def swat_one_liner(data_dir):
    train_data = pd.read_csv(f'{data_dir}/SWAT/swat_train.csv').values
    test_data = pd.read_csv(f'{data_dir}/SWAT/swat_test.csv').values
    labels = pd.read_csv(f'{data_dir}/SWAT/labels_test.csv').values
    len_test = len(test_data)

    train_means = train_data.mean(axis=0, keepdims=True)

    train_errors = np.square(train_data-train_means)
    train_errors = train_errors.mean(axis=0)
    train_errors = train_errors[None, :] + 0.01

    scores = np.abs(test_data - train_means)
    scores = scores[len_test//2:]
    scores = scores/(train_errors)
    scores[:, 10] = 0
    scores = scores.mean(axis=1)
    labels = labels[len_test//2:, 0]

    f1, precision, recall, *_ = best_f1_linspace(scores=scores, labels=labels, n_splits=100, segment_adjust=False)

    return f1, precision, recall

def swat_nn(data_dir):
    downsampling_size = 10
    
    train_data = pd.read_csv(f'{data_dir}/SWAT/swat_train.csv').values
    train_data = train_data[:,None,:]
    test_data = pd.read_csv(f'{data_dir}/SWAT/swat_test.csv').values
    test_data = test_data[:,None,:]
    labels = pd.read_csv(f'{data_dir}/SWAT/labels_test.csv').values
    len_test = len(labels)
    
    # Padding
    dataset = np.vstack([train_data, test_data])

    right_padding = downsampling_size - dataset.shape[0]%downsampling_size
    dataset = np.pad(dataset, ((right_padding, 0), (0,0), (0,0) ))

    right_padding = downsampling_size - len_test%downsampling_size

    dataset = dataset.reshape(dataset.shape[0]//downsampling_size, -1, 1, dataset.shape[2]).max(axis=1)
    len_test_downsampled = int(np.ceil(len_test/downsampling_size))

    train_data = dataset[:-len_test_downsampled, 0, :]
    test_data = dataset[-len_test_downsampled:, 0, :]

    train_max = train_data.max(axis=0, keepdims=True)
    train_data = train_data/train_max
    test_data = test_data/train_max

    nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(train_data)
    distances, _ = nbrs.kneighbors(test_data)
    scores = distances.mean(axis=1)

    labels = labels[len_test//2:, 0]
    scores = scores.repeat(10)[-len(labels):]

    f1, precision, recall, *_ = best_f1_linspace(scores=scores, labels=labels, n_splits=100, segment_adjust=False)

    return f1, precision, recall