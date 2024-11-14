# ISC 24 - Tutorial on Distributed Training of Deep Neural Networks

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
sbatch run.sh -f configs/single_gpu.json
```

### Mixed Precision

```bash
sbatch run.sh -f configs/single_gpu_mp.json
```


## Data Parallelism

### Pytorch Distributed Data Parallel (DDP)

```bash
sbatch run.sh -f configs/ddp.json
```

### Fully Sharded Data Parallelism (FSDP)


```bash
sbatch run.sh -f configs/fsdp.json
```

## Tensor Parallelism

```bash
sbatch run.sh -f configs/axonn.json
```

## Inference

```bash
```
