#!/bin/bash
#SBATCH --job-name=gemma_unsloth_grpo_no4bit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/gemma_unsloth_grpo_no4bit_%j.log
#SBATCH --error=logs/gemma_unsloth_grpo_no4bit_%j.err

mkdir -p logs
mkdir -p outputs/gemma_e2b_grpo_unsloth_no4bit

source ~/.bashrc
conda activate /mmfs1/scratch/jacks.local/pkhanal2568/envs/gemma4_grpo/

echo "Python:"
which python

echo "Torch version:"
python -c "import torch; print(torch.__version__)"

echo "Transformers version:"
python -c "import transformers; print(transformers.__version__)"

echo "Checking GPU:"
nvidia-smi

echo "Checking Unsloth:"
python -c "import unsloth; from unsloth import FastVisionModel; print('Unsloth import OK')"

export USE_UNSLOTH=1
export LOAD_IN_4BIT=0
export LOG_COMPLETION_DETAILS=0
export RUN_NAME=gemma_e2b_grpo_unsloth_no4bit
export OUTPUT_DIR=outputs/gemma_e2b_grpo_unsloth_no4bit

python training/grpo_train_with_logging.py