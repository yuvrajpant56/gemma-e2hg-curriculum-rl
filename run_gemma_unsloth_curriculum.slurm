#!/bin/bash
#SBATCH --job-name=gemma_unsloth_grpo_curriculum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/gemma_unsloth_grpo_curriculum_%j.log
#SBATCH --error=logs/gemma_unsloth_grpo_curriculum_%j.err

set -e

# -----------------------------
# 1. Go to project directory
# -----------------------------
cd /mmfs1/scratch/jacks.local/pkhanal2568/Gemma_Hackathon/Grpo_Training

# -----------------------------
# 2. Create log/output folders
# -----------------------------
mkdir -p logs
mkdir -p outputs/gemma_e2b_grpo_unsloth_curriculum_v1

# -----------------------------
# 3. Activate conda environment
# -----------------------------
source ~/.bashrc
conda activate /mmfs1/scratch/jacks.local/pkhanal2568/envs/gemma4_grpo/

echo "======================================"
echo "Job started on: $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo "======================================"

echo "Python:"
which python

echo "Transformers version:"
python -c "import transformers; print(transformers.__version__)"

echo "Torch / CUDA check:"
python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo "GPU:"
nvidia-smi

echo "Checking Unsloth:"
python -c "from unsloth import FastVisionModel; print('Unsloth import OK')"

# -----------------------------
# 4. Run GRPO with curriculum sampler + logging
# -----------------------------
USE_UNSLOTH=1 \
LOAD_IN_4BIT=1 \
LOG_COMPLETION_DETAILS=1 \
RUN_NAME=gemma_e2b_grpo_unsloth_curriculum \
OUTPUT_DIR=outputs/gemma_e2b_grpo_unsloth_curriculum_v1 \
DATA_SCHEDULE=gaussian \
CURRICULUM_BETA=0.5 \
CURRICULUM_SIGMA=0.5 \
CURRICULUM_MIN_PROB=0.0 \
DEBUG_SAMPLER=1 \
DEBUG_SAMPLER_EVERY=5 \
python training/grpo_train_with_logging_curriculum.py

echo "======================================"
echo "Job finished on: $(date)"
echo "======================================"