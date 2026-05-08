import os
import sys
import random

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from huggingface_hub import login
from typing import Any, Dict, List
from peft import LoraConfig, TaskType, get_peft_model
from config import USE_LORA, LORA_CONFIG
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

from config import (
    MODEL_NAME,
    TRAIN_DATASET_PATHS,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    TOP_P,
    DO_SAMPLE,
    DEVICE_MAP,
    TORCH_DTYPE,
    HF_TOKEN,
)
from dataset.loader import load_dataset
from prompting.prompt_builder import CountdownPromptBuilder
from rewards.reward_fn import CountdownRewardFunction


def build_hf_dataset(
    tokenizer,
    dataset_paths: List[str],
    max_samples_per_dataset: int | None = None,
    balance_buckets: bool = True,
    seed: int = 42,
) -> Dataset:
    """
    Build one Hugging Face Dataset from multiple parquet files.

    Each dataset path is loaded separately.
    If balance_buckets=True, each difficulty bucket is downsampled to the
    smallest bucket size so the mixed baseline is not dominated by one file.
    """
    prompt_builder = CountdownPromptBuilder()
    all_bucket_rows = []

    for dataset_path in dataset_paths:
        raw_rows = load_dataset(dataset_path, max_samples=max_samples_per_dataset)
        print(f"Loaded {len(raw_rows)} rows from: {dataset_path}")
        all_bucket_rows.append(raw_rows)

    if len(all_bucket_rows) == 0:
        raise ValueError("No datasets were loaded. Check TRAIN_DATASET_PATHS in config.py")

    if balance_buckets:
        non_empty_sizes = [len(rows) for rows in all_bucket_rows if len(rows) > 0]
        if len(non_empty_sizes) == 0:
            raise ValueError("All loaded datasets are empty.")
    
        min_size = min(non_empty_sizes)
        print(f"Balancing enabled. Using {min_size} samples from each bucket.")
    
        balanced_bucket_rows = []
        for bucket_idx, rows in enumerate(all_bucket_rows):
            rows_copy = rows[:]
            bucket_rng = random.Random(seed + bucket_idx)
            bucket_rng.shuffle(rows_copy)
            balanced_bucket_rows.append(rows_copy[:min_size])
    else:
        print("Balancing disabled. Using all samples from all buckets.")
        balanced_bucket_rows = all_bucket_rows

    processed_rows = []
    for bucket_rows in balanced_bucket_rows:
        for row in bucket_rows:
            prompt_example = prompt_builder.build_example_for_prompt(row)
            prompt_data = prompt_builder.generate_prompt(tokenizer, prompt_example)

            processed_rows.append(
                {
                    "prompt": prompt_data["prompt"],
                    "target": row["target"],
                    "numbers": row["numbers"],
                    "difficulty_n": row["difficulty_n"],
                    "split": row["split"],
                    "sample_idx": row["sample_idx"],
                }
            )

    rng = random.Random(seed)
    rng.shuffle(processed_rows)

    print(f"Final mixed training dataset size: {len(processed_rows)}")
    return Dataset.from_list(processed_rows)


def _unwrap_completion_text(completion: Any) -> str:
    """
    TRL passes completions differently depending on dataset format.

    Official docs:
    - standard format -> completions are strings
    - conversational format -> completions are message dicts

    This helper makes the reward function robust to either case.
    """
    if isinstance(completion, str):
        return completion

    if isinstance(completion, list):
        # conversational-style completion like:
        # [{"role": "assistant", "content": "..."}]
        if len(completion) > 0 and isinstance(completion[0], dict):
            return completion[0].get("content", "")
        # fallback
        return " ".join(str(x) for x in completion)

    if isinstance(completion, dict):
        return completion.get("content", "")

    return str(completion)


def countdown_grpo_reward(
    prompts,
    completions,
    target,
    numbers,
    trainer_state=None,
    **kwargs,
) -> List[float]:
    """
    Custom reward function for GRPO.

    TRL passes:
    - prompts
    - completions
    - any extra dataset columns such as target and numbers

    We keep allow_implicit_think_open=True to match your current setup.
    """
    reward_engine = CountdownRewardFunction(
        tolerance=1e-6,
        allow_implicit_think_open=True,
    )

    completion_texts = [_unwrap_completion_text(c) for c in completions]
    
    n_prompts = len(prompts)
    n_completions = len(completion_texts)
    n_targets = len(target)
    n_numbers = len(numbers)

    if not (n_prompts == n_completions == n_targets == n_numbers):
        print("PROMPT EXAMPLE:", repr(prompts[0]) if len(prompts) > 0 else "NONE")
        print("COMPLETION EXAMPLE:", repr(completion_texts[0]) if len(completion_texts) > 0 else "NONE")
        print("TARGET EXAMPLE:", repr(target[0]) if len(target) > 0 else "NONE")
        print("NUMBERS EXAMPLE:", repr(numbers[0]) if len(numbers) > 0 else "NONE")
        raise ValueError(
            f"Alignment mismatch in reward function: "
            f"prompts={n_prompts}, completions={n_completions}, "
            f"target={n_targets}, numbers={n_numbers}"
        )

    rewards: List[float] = []
    
    for i in range(n_completions):
        completion_text = completion_texts[i]
        target_i = target[i]
        numbers_i = numbers[i]

        result = reward_engine.score_single(
            completion_text=completion_text,
            prompt_numbers=numbers_i,
            target=target_i,
        )
        rewards.append(float(result["reward"]))

    return rewards
    

class PatchedGRPOTrainer(GRPOTrainer):
    def _get_train_sampler(self, train_dataset=None):
        # Accept and ignore the extra argument so _get_dataloader can call
        # sampler_fn(dataset) without crashing.
        return super()._get_train_sampler()
        



def build_lora_config() -> LoraConfig:
    return LoraConfig(
        r=LORA_CONFIG["r"],
        lora_alpha=LORA_CONFIG["lora_alpha"],
        lora_dropout=LORA_CONFIG["lora_dropout"],
        target_modules=LORA_CONFIG["target_modules"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )


def main() -> None:
    # Safer than hardcoding in a shared script:
    hf_token = os.environ.get("HF_TOKEN", HF_TOKEN)
    print(f"hf_token={hf_token}")
    if not hf_token:
        raise ValueError("Set HF_TOKEN in your environment or config.py")
        
    # Make token available to all Hugging Face/PEFT internal calls
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
    
    #log in explicitly so internal hub lookups also use the token
    login(token=hf_token, add_to_git_credential=False)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        token=hf_token,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=hf_token,
        trust_remote_code=True,
        torch_dtype="auto" if TORCH_DTYPE == "auto" else getattr(torch, TORCH_DTYPE),
        device_map=DEVICE_MAP,
    )
    
    if USE_LORA:
        lora_config = build_lora_config()
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Start small for debugging
    train_dataset = build_hf_dataset(
        tokenizer=tokenizer,
        dataset_paths=TRAIN_DATASET_PATHS,
        max_samples_per_dataset=None,
        balance_buckets=True,
        seed=42,
    )

    # TRL requires the effective batch size:
    # num_processes * per_device_train_batch_size * gradient_accumulation_steps
    # to be evenly divisible by num_generations. :contentReference[oaicite:1]{index=1}
    
    training_args = GRPOConfig(
        output_dir="outputs/grpo_countdown_mixed_n2_n5_baseline",
        run_name="grpo_countdown_mixed_n2_n5_baseline",
        
        logging_steps=5,
        save_steps=100,
        save_total_limit=2,
        
        max_steps=1600,
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        
        
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_generations=4,
        
        seed=42,
        beta=0.001,
        max_grad_norm=1.0,
    
        max_prompt_length=1024,
        max_completion_length=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        
        log_completions=True,
        report_to="none",
        remove_unused_columns=False,
        
        bf16=(TORCH_DTYPE == "bfloat16") or (
            TORCH_DTYPE == "auto" and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        ),
        fp16=(TORCH_DTYPE == "float16"),
    )

    trainer = PatchedGRPOTrainer(
    model=model,
    reward_funcs=[countdown_grpo_reward],
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()