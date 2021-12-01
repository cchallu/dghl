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
import argparse
import glob
import pandas as pd

from utils.utils_data import load_smd, load_nasa, load_swat
from utils.utils_experiment import train_DGHL

def basic_mc(dataset, random_seed):
    if dataset == 'MSL':
        n_features = 55
        normalize_windows = False
    elif dataset == 'SMAP':
        n_features = 25
        normalize_windows = True
    elif dataset == 'SMD':
        n_features = 38
        normalize_windows = False
    elif dataset == 'SWAT':
        n_features = 51
        normalize_windows = False     

    mc = {}
    mc['window_size'] = 64*4 # 64
    mc['window_step'] = 64*4
    mc['n_features'] = n_features
    mc['hidden_multiplier'] = 32
    mc['max_filters'] = 256
    mc['kernel_multiplier'] = 1
    mc['z_size'] = 20
    mc['z_size_up'] = 5
    mc['window_hierarchy'] = 1 #4
    mc['z_iters'] = 25
    mc['z_sigma'] = 0.25
    mc['z_step_size'] = 0.1
    mc['z_with_noise'] = False
    mc['z_persistent'] = True
    mc['z_iters_inference'] = 500
    mc['batch_size'] = 4
    mc['learning_rate'] = 1e-3
    mc['noise_std'] = 0.001
    mc['n_iterations'] = 1000
    mc['normalize_windows'] = normalize_windows
    mc['random_seed'] = random_seed
    mc['device'] = None

    return mc

def run_smd(args):
    mc = basic_mc(dataset='SMD', random_seed=args.random_seed)

    # LOOP MACHINES
    files = sorted(glob.glob('./data/ServerMachineDataset/train/*.txt'))
    for file in files:
        machine = file[:-4].split('/')[-1]
        print(50*'-', machine, 50*'-')

        root_dir = f'./results/{args.experiment_name}_{args.random_seed}/SMD'
        os.makedirs(name=root_dir, exist_ok=True)

        train_data, train_mask, test_data, test_mask, labels = load_smd(entities=machine, downsampling_size=10,
                                                                        occlusion_intervals=args.occlusion_intervals,
                                                                        occlusion_prob=args.occlusion_prob,
                                                                        root_dir='./data', verbose=True)
        
        train_DGHL(mc=mc, train_data=train_data, test_data=test_data, test_labels=labels,
                   train_mask=train_mask, test_mask=test_mask, entities=[machine], make_plots=True, root_dir=root_dir)

def run_smap(args):
    mc = basic_mc(dataset='SMAP', random_seed=args.random_seed)

    # LOOP CHANNELS
    meta_data = pd.read_csv('./data/NASA/labeled_anomalies.csv')
    entities = meta_data[meta_data['spacecraft']=='SMAP']['chan_id'].values
    for entity in entities:
        print(50*'-', entity, 50*'-')

        root_dir = f'./results/{args.experiment_name}_{args.random_seed}/SMAP'
        os.makedirs(name=root_dir, exist_ok=True)

        train_data, train_mask, test_data, test_mask, labels = load_nasa(entities=[entity], downsampling_size=1,
                                                                         occlusion_intervals=args.occlusion_intervals,
                                                                         occlusion_prob=args.occlusion_prob,
                                                                         root_dir='./data', verbose=True)

        train_DGHL(mc=mc, train_data=train_data, test_data=test_data, test_labels=labels,
                   train_mask=train_mask, test_mask=test_mask, entities=[entity], make_plots=True, root_dir=root_dir)

def run_msl(args):
    mc = basic_mc(dataset='MSL', random_seed=args.random_seed)

     # LOOP CHANNELS
    meta_data = pd.read_csv('./data/NASA/labeled_anomalies.csv')
    entities = meta_data[meta_data['spacecraft']=='MSL']['chan_id'].values
    for entity in entities:
        print(50*'-', entity, 50*'-')

        root_dir = f'./results/{args.experiment_name}_{args.random_seed}/MSL'
        os.makedirs(name=root_dir, exist_ok=True)

        train_data, train_mask, test_data, test_mask, labels = load_nasa(entities=[entity], downsampling_size=1,
                                                                         occlusion_intervals=args.occlusion_intervals,
                                                                         occlusion_prob=args.occlusion_prob,
                                                                         root_dir='./data', verbose=True)

        train_DGHL(mc=mc, train_data=train_data, test_data=test_data, test_labels=labels,
                   train_mask=train_mask, test_mask=test_mask, entities=[entity], make_plots=True, root_dir=root_dir)

def run_swat(args):
    mc = basic_mc(dataset='SWAT', random_seed=args.random_seed)

    root_dir = f'./results/{args.experiment_name}_{args.random_seed}/SWAT'
    os.makedirs(name=root_dir, exist_ok=True)

    train_data, train_mask, test_data, test_mask, labels = load_swat(downsampling_size=10,
                                                                     occlusion_intervals=args.occlusion_intervals,
                                                                     occlusion_prob=args.occlusion_prob,
                                                                     root_dir='./data', verbose=True)
    
    train_DGHL(mc=mc, train_data=train_data, test_data=test_data, test_labels=labels,
               train_mask=train_mask, test_mask=test_mask, entities=['SWAT'], make_plots=True, root_dir=root_dir)

def main(args):
    print(105*'-')
    print(50*'-',' SMD ', 50*'-')
    print(105*'-')
    run_smd(args)

    print(106*'-')
    print(50*'-',' SMAP ', 50*'-')
    print(106*'-')
    run_smap(args)

    print(105*'-')
    print(50*'-',' MSL ', 50*'-')
    print(105*'-')
    run_msl(args)

    print(105*'-')
    print(50*'-',' SWAT ', 50*'-')
    print(105*'-')
    run_swat(args)

def parse_args():
    desc = "Run DGHL in benchmark datasets"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--random_seed', type=int, required=True)
    parser.add_argument('--occlusion_intervals', type=int, required=True)
    parser.add_argument('--occlusion_prob', type=float, required=True)
    parser.add_argument('--experiment_name', type=str, required=True)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if args is None:
        exit()
    
    main(args)

# source ~/anaconda3/etc/profile.d/conda.sh
# CUDA_VISIBLE_DEVICES=2 python src/run_dghl.py --random_seed 1 --occlusion_intervals 1 --occlusion_prob 0 --experiment_name 'DGHL'