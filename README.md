# SC 24 - Tutorial on Distributed Training of Deep Neural Networks

[![Join slack](https://img.shields.io/badge/slack-axonn--users-blue)](https://join.slack.com/t/axonn-users/shared_invite/zt-2itbahk29-_Ig1JasFxnuVyfMtcC4GnA)

All the code for the hands-on exercies can be found in this repository. 

**Table of Contents**

* [Setup](#setup)
* [Basics of Model Training](#basics-of-model-training)
* [Data Parallelism](#data-parallelism)
* [Tensor Parallelism](#tensor-parallelism)
* [Inference](#inference)

## Setup 

To request an account on Zaratan, please join slack at the link above, and fill [this Google form]().

We have pre-built the dependencies required for this tutorial on Zaratan. This
will be activated automatically when you run the bash scripts.

Model weights and the training dataset have 
been downloaded in `/scratch/zt1/project/isc/shared/`.

## Basics of Model Training

### Using PyTorch Lightning

```bash
CONFIG_FILE=configs/single_gpu.json sbatch --ntasks-per-node=1  train.sh
```

### Mixed Precision
Open `configs/single_gpu.json` and change `precision` to `bf16-mixed` and then run - 

```bash
CONFIG_FILE=configs/single_gpu.json sbatch --ntasks-per-node=1  train.sh
```


## Data Parallelism

### Pytorch Distributed Data Parallel (DDP)

```bash
CONFIG_FILE=configs/ddp.json sbatch --ntasks-per-node=4  train.sh
```

### Fully Sharded Data Parallelism (FSDP)


```bash
CONFIG_FILE=configs/fsdp.json sbatch --ntasks-per-node=4  train.sh
```

## Tensor Parallelism

```bash
CONFIG_FILE=configs/axonn.json sbatch --ntasks-per-node=4  train.sh
```

## Inference

Add more prompts to `data/inference/prompts.txt` if you want. Then run

```bash
CONFIG_FILE=configs/inference_axonn.json sbatch --ntasks-per-node=1  infer.sh
```

### With torch.compile

Open `configs/axonn_inference.json` and change `compile` to `true`. Then run 

```bash
CONFIG_FILE=configs/inference_axonn.json sbatch --ntasks-per-node=1  infer.sh
```

### With tensor parallelism

Open `configs/axonn_inference.json` and change `tp_dimensions` to `[4, 1, 1]`. Then run

```bash
CONFIG_FILE=configs/inference_axonn.json sbatch --ntasks-per-node=4  infer.sh
```
