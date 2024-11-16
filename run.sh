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

echo "Copying python environment to fast node local storage"
start=`date +%s`
#mkdir -p /tmp/tutorial_env
#tar -xzf ${SCRATCH}/miniconda3.tar.gz -C /tmp/tutorial_env
end=`date +%s`
runtime=$((end-start))
echo "Copy completed. Time taken = ${runtime} s"

# activate environment
source /tmp/tutorial_env/bin/activate


CONFIG_FILE="${CONFIG_FILE:-"configs/single_gpu.json"}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"


export OMP_NUM_THREADS=16
export PYTHONUNBUFFERED=1

# Run torchrun with specified number of GPUs
torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=1 --node_rank=0 train.py --config-file $CONFIG_FILE

