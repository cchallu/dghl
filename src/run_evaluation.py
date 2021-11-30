import numpy as np
import argparse

from utils.utils_evaluation import smd_compute_f1, nasa_compute_f1

def main(args):
    # SMD
    print(10*'-', 'SMD', 10*'-')
    smd_scores = []
    for i in range(1, 5):
        experiment = f'{args.experiment_name}_{i}'
        f1, _, _ = smd_compute_f1(scores_dir=f'./results/{experiment}', n_splits=100, data_dir='./data')
        print(f1)
        smd_scores.append(f1)

    print(100*np.mean(smd_scores))
    print(100*np.std(smd_scores))

    # SMAP
    print(10*'-', 'SMAP', 10*'-')
    smap_scores = []
    for i in range(1, 5):
        experiment = f'{args.experiment_name}_{i}'
        f1, _, _ = nasa_compute_f1(dataset='SMAP', scores_dir=f'./results/{experiment}', n_splits=100, data_dir='./data')
        print(f1)
        smap_scores.append(f1)
    
    print(100*np.mean(smap_scores))
    print(100*np.std(smap_scores))

    # MSL
    print(10*'-', 'MSL', 10*'-')
    msl_scores = []
    for i in range(1, 5):
        experiment = f'{args.experiment_name}_{i}'
        f1, _, _ = nasa_compute_f1(dataset='MSL', scores_dir=f'./results/{experiment}', n_splits=100, data_dir='./data')
        print(f1)
        msl_scores.append(f1)
    
    print(100*np.mean(msl_scores))
    print(100*np.std(msl_scores))

def parse_args():
    desc = "Run DGHL in benchmark datasets"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--experiment_name', type=str, required=True)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if args is None:
        exit()
    
    main(args)


# python src/run_evaluation.py --experiment_name 'encoder_gpu'