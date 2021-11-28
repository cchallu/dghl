# DGHL

This repo provides an implementation of the DGHL model and produces the results for the main table presented in the paper.

# Data
The datasets are not included in the repository, but are available in the following links:

### NASA
SMAP and MSL datasets are available at https://s3-us-west-2.amazonaws.com/telemanom/data.zip. Labels are available at https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv. Refer to https://github.com/khundman/telemanom for more details of these datasets.

### SMD
SMD dataset is available at https://github.com/NetManAIOps/OmniAnomaly.


# Environment
Run `setup.sh` to create a conda environment with the packages needed to run the experiments (this file does not consider different CUDA version to run in GPU. Refer to https://pytorch.org/get-started/locally/ for PyTorch installation).

# Run DGHL experiment from console
To replicate the results of the paper, in particular to produce the anomaly scores for DGHL, run the `run_dghl.py' script:

```console
python src/run_dghl.py --random_seed 1 --experiment_name 'DGHL'
```

The random_seed argument specifies the random seed for the model parameters initialization and Langevin Dynamics initial state. To replicate results of the paper, run with seeds 1-4. Experiment name can be used to identify different runs. This script will produce anomaly scores for the four datasets considered in the paper. 

# Evaluation

The evaluation functions are included in `utils_evaluation.py`. Use the notebook `analyze_complete_run.ipynb` to run the evaluation. You might need to modify the directory of the results. The notebook `simple_benchmarks.ipynb` runs simple benchmarks, Mean Deviation and Nearest Neighbors, on the four datasets.


