#!/bin/bash
#SBATCH --job-name=cpm-appraisal
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=5G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j.out

module load miniconda3
conda activate ssp

export HF_HOME=/scratch/$USER/hf_cache   # match what you used when downloading

cd $HOME/cpm-appraisal

python scripts/run_experiment.py \
    --backend local_transformers \
    --model-id Qwen/Qwen2.5-7B-Instruct \
    --out data/outputs/results_qwen.json
