#!/bin/bash
#SBATCH -N 1
#SBATCH -t 00:03:00
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:4
#SBATCH -A sc24-class
#SBATCH --ntasks-per-node 128
#SBATCH --exclusive
#SBATCH --mem=500G

export SCRATCH="/scratch/zt1/project/sc24/shared/"
export HF_HOME="${SCRATCH}/.cache/huggingface"
export HF_TRANSFORMERS_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

module load python
source $SCRATCH/axonn_venv/bin/activate

# Default values
ARGS_FILE="configs/single_gpu.json"
GPUS=1

# Parse command-line arguments
while getopts g:f: flag
do
    case "${flag}" in
        g) GPUS=${OPTARG};;
        f) ARGS_FILE=${OPTARG};;
        *) echo "Invalid option"; exit 1;;
    esac
done

SCRIPT="train.py --file $ARGS_FILE"


export OMP_NUM_THREADS=16
export PYTHONUNBUFFERED=1
# Run torchrun with specified number of GPUs
torchrun --nproc_per_node=$GPUS --nnodes=1 --node_rank=0 $SCRIPT

