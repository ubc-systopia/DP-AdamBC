# Repository Information

Modified version of:

A JAX implementation of ['Node-Level Differentially Private Graph
Neural Networks'](https://github.com/google-research/google-research/tree/master/differentially_private_gnns), presented at PAIR2Struct, ICLR 2022.



### Instructions

Create and activate a virtual environment:

```shell
python -m venv .venv && source .venv/bin/activate
```

Install dependencies with:

```shell
pip install --upgrade pip && pip install --upgrade "jax[cuda12_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && pip install -r requirements.txt
```

Download a dataset (ogbn-arxiv shown below) with our script:

```shell
python download_datasets.py --dataset_name=ogbn-arxiv
```

Start training with a configuration defined
under `configs/`:

```shell
python main.py --workdir=./tmp --config=configs/dpgcn.py
```

#### Changing Hyperparameters

Since the configuration is defined using
[config_flags](https://github.com/google/ml_collections/tree/master#config-flags),
you can override hyperparameters. For example, to change the number of training
steps, the batch size and the dataset:

```shell
python main.py --workdir=./tmp --config=configs/dpgcn.py \
--config.num_training_steps=10 --config.batch_size=50 \
--config.dataset=reddit-disjoint
```

For more extensive changes, you can directly edit the configuration files,
and even add your own.

#### Changing Dataset Paths

By default, the datasets are downloaded to `datasets/` in the current working
directory. You can change this by passing `dataset_path` to the download script:

```shell
python download_datasets.py --dataset_name=ogbn-arxiv \
--dataset_path=${DATASET_PATH}
```

and then updating the path in the config:

```shell
python main.py --workdir=./tmp --config=configs/dpgcn.py \
--config.dataset_path=${DATASET_PATH}
```

#### Tuning Hyperparameters

We use [Optuna](https://optuna.readthedocs.io/en/stable/) to tune hyperparameters 
automatically. You can run a tuning process using `autotune.py` over 50 runs
(hyperparameter combinations):

```shell
python autotune.py --in_path tune_adambc_example.json --n 50 \
--workdir=./tmp --config=configs/dpgcn.py
```

The filed that can be used for tuning are stored in `autotune/`, you can specify
which hyperparameters should be constant, and over which ranges and value
hyperparameters should be tuned. After each run, the best hyperparameters are stored
stored in `autotune/best_params/`. The tuning process is also stored in SQL in
`autotune/sql/`, so it can be continued later if necessary.

### Notes

This is a simpler and faster JAX implementation
that differs from the TensorFlow implementation
used to obtain results in the paper.
The main constraint is because XLA requires
fixed-size arrays to represent subgraphs.

### Citation

Please cite the original authors if you use this code :)

```text
@inproceedings{
daigavane2022nodelevel,
title={Node-Level Differentially Private Graph Neural Networks},
author={Ameya Daigavane and Gagan Madan and Aditya Sinha and Abhradeep Guha Thakurta and Gaurav Aggarwal and Prateek Jain},
booktitle={ICLR 2022 Workshop on PAIR{\textasciicircum}2Struct: Privacy, Accountability, Interpretability, Robustness, Reasoning on Structured Data},
year={2022},
url={https://openreview.net/forum?id=BCfgOLx3gb9}
}
```

