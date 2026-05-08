#!/bin/bash
#SBATCH --job-name=gemma_grpo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/gemma_grpo_%j.log
#SBATCH --error=logs/gemma_grpo_%j.err

# Load required modules
#module load <module name>

# Activate your environment if needed
source ~/.bashrc
conda activate /mmfs1/scratch/jacks.local/pkhanal2568/envs/gemma4_grpo/


mkdir -p logs
mkdir -p outputs/gemma_e2b_grpo_standard_v5

USE_UNSLOTH=0 \
RUN_NAME=gemma_e2b_grpo_standard \
OUTPUT_DIR=outputs/gemma_e2b_grpo_standard_v5 \
python training/grpo_train_with_logging.py