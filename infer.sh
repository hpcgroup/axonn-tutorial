#!/bin/bash
#SBATCH -N 1
#SBATCH -t 00:05:00
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:4
#SBATCH -A sc24-class
#SBATCH --exclusive
#SBATCH --mem=500G


export SCRATCH="/scratch/zt1/project/sc24/shared/"
export HF_HOME="${SCRATCH}/.cache/huggingface"
export HF_TRANSFORMERS_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500


echo "Copying python environment to fast node local storage"
start=`date +%s`
mkdir -p /tmp/tutorial_env
tar -xzf ${SCRATCH}/miniconda3.tar.gz -C /tmp/tutorial_env
end=`date +%s`
runtime=$((end-start))
echo "Copy completed. Time taken = ${runtime} s"

# activate environment
source /tmp/tutorial_env/bin/activate

CONFIG_FILE="${CONFIG_FILE:-configs/inference_axonn.json}"
echo $CONFIG_FILE

# Run torchrun with specified number of GPUs
#srun -u python -u infer.py --config-file $CONFIG_FILE
srun -u python -u infer.py --config-file $CONFIG_FILE
