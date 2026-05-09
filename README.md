---
license: apache-2.0
base_model: google/gemma-4-E4B-it
tags:
- gemma
- grpo
- reinforcement-learning
- curriculum-learning
- easy-to-hard
- countdown
- reasoning
- lora
- unsloth
language:
- en
---

# Gemma E2H-G Countdown unsloth GRPO LoRA

## Project Motivation

Curriculum learning is a training strategy where a model learns from easier examples first and gradually moves toward harder examples. This idea is inspired by how humans often learn: we usually begin with simple problems, build basic skills, and then use those skills to solve more difficult tasks.

The motivation for this project comes from the recent research paper **“Curriculum Reinforcement Learning from Easy to Hard Tasks Improves LLM Reasoning,” published as a conference paper at ICLR 2026**. The main idea of that work is that reinforcement learning can improve the reasoning ability of language models when the training process is organized from easy tasks to hard tasks, instead of exposing the model to all task difficulties randomly from the beginning.

In reasoning tasks, harder examples often provide sparse rewards because the model may fail completely and receive little useful learning signal. Easy-to-hard curriculum learning helps address this issue by first allowing the model to learn basic reasoning patterns from simpler problems. Once these easier patterns are learned, the model can gradually transfer them to more complex problems.

This repository applies that idea to a small Gemma model using Countdown arithmetic reasoning tasks.

## Overview

This repository contains a LoRA adapter fine-tuned with curriculum-guided reinforcement learning for Countdown arithmetic reasoning.

The goal is to test whether a small Gemma model can improve its reasoning ability when trained with an easy-to-hard curriculum. Instead of training only with randomly mixed tasks, the model is gradually exposed to problems with increasing difficulty.

The task is simple but strict: given a target number and a list of numbers, the model must generate an arithmetic expression that evaluates exactly to the target while using each provided number exactly once.

## Base Model

`google/gemma-4-E4B-it`

## Training Method

The model was fine-tuned using GRPO-style reinforcement learning with LoRA adapters through Unsloth.

The curriculum is based on the number of operands in the Countdown task:

- `n2`: two-number tasks
- `n3`: three-number tasks
- `n4`: four-number tasks
- `n5`: five-number tasks

The core hypothesis is:

> If the model first learns easier arithmetic reasoning patterns, then it can transfer those patterns to harder multi-step reasoning tasks.

## Reward Design

The reward function checks whether the model follows both the reasoning format and the arithmetic constraints.

The reward validates:

1. Correct output format using `<think>...</think>` and `<answer>...</answer>` tags.
2. Arithmetic syntax validity.
3. Use of only the provided numbers.
4. Use of each provided number exactly once.
5. Exact match between the final expression and the target value.

This makes the task verifiable because the final answer can be automatically checked.

## Evaluation Results

The curriculum-trained LoRA model was compared against the base Gemma model on 400 held-out Countdown examples across four difficulty levels.

| Difficulty | Base Accuracy | Curriculum LoRA Accuracy | Improvement |
|---|---:|---:|---:|
| n2 | 100% | 100% | 0 pts |
| n3 | 82% | 87% | +5 pts |
| n4 | 38% | 43% | +5 pts |
| n5 | 1% | 9% | +8 pts |
| Overall | 55.25% | 59.75% | +4.50 pts |

The curriculum-trained LoRA improves overall exact accuracy by **+4.5 percentage points**.

The strongest improvement appears on the hardest `n5` split, where accuracy improves from **1% to 9%**. This suggests that curriculum-guided reinforcement learning helps the small Gemma model transfer reasoning behavior from easier tasks to more difficult arithmetic problems.

## Key Finding

The base model already performs well on easier `n2` tasks, but its performance drops sharply as the number of operands increases. After curriculum-guided GRPO training, the model maintains perfect performance on `n2` while improving on `n3`, `n4`, and especially `n5`.

This supports the main idea behind easy-to-hard curriculum reinforcement learning:

> Small language models can improve reasoning performance when reinforcement learning is structured around a gradual progression from simple to complex tasks.

## Intended Use

This adapter is intended for research and demonstration purposes in:

- Small language model reasoning
- Curriculum learning
- Reinforcement learning from verifiable rewards
- GRPO-style post-training
- Countdown arithmetic reasoning
- LoRA-based efficient fine-tuning

## Limitations

The model still struggles with harder `n4` and `n5` problems. Many failures occur because the model begins a reasonable solution path but generates overly long reasoning and fails to close the required `<think>` or `<answer>` tags before generation ends.

This suggests that future work should improve:

- Length control
- Format-following rewards
- Concise reasoning generation
- Multi-seed evaluation
- Adaptive curriculum scheduling
- Answer-first or verifier-guided decoding

This adapter should be viewed as a research artifact rather than a production-ready math solver.

## How to Train the Model

### 1. Clone the repository

```bash
git clone https://github.com/yuvrajpant56/gemma-e2hg-curriculum-rl.git
cd gemma-e2hg-curriculum-rl
conda create -n gemma_grpo python=3.10 -y
conda activate gemma_grpo
pip install -r requirements.txt
export HF_TOKEN="your_huggingface_token_here"
huggingface-cli login
python training/grpo_train_with_logging_curriculum.py
```


## Citation / Inspiration

This project is inspired by:

**Curriculum Reinforcement Learning from Easy to Hard Tasks Improves LLM Reasoning**  
Shubham Parashar, Shurui Gui, Xiner Li, Hongyi Ling, Sushil Vemuri, Blake Olson, Eric Li, Yuhang James Caverlee, Dileep Kalathil, Shuiwang Ji  
Published as a conference paper at ICLR 2026.

EOF
