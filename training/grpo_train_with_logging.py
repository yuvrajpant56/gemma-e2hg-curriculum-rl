"""
Base GRPO training script with optional Unsloth loading and experiment logging.

This keeps your existing project structure:
- config.py
- dataset/loader.py
- prompting/prompt_builder.py
- rewards/reward_fn.py

It adds:
1) optional Unsloth model loading through USE_UNSLOTH=True in config.py or env USE_UNSLOTH=1
2) dataset snapshots for reproducibility
3) bucket-count CSV for write-up tables
4) reward-detail JSONL for later analysis
5) trainer metric CSV/JSON for plotting reward/loss/learning-rate curves
"""

import csv
import json
import os
import random
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    
USE_UNSLOTH_EARLY = os.environ.get("USE_UNSLOTH", "0").lower() in {"1", "true", "yes", "y"}

if USE_UNSLOTH_EARLY:
    import unsloth
    from unsloth import FastVisionModel, PatchFastRL
    PatchFastRL("GRPO", FastVisionModel)

import torch
from datasets import Dataset
from huggingface_hub import login
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from config import (
    DEVICE_MAP,
    DO_SAMPLE,
    HF_TOKEN,
    LORA_CONFIG,
    MAX_NEW_TOKENS,
    MODEL_NAME,
    TEMPERATURE,
    TOP_P,
    TORCH_DTYPE,
    TRAIN_DATASET_PATHS,
    USE_LORA,
)

# Optional config values. These do not break your old config.py if absent.
try:
    from config import USE_UNSLOTH as CONFIG_USE_UNSLOTH
except Exception:
    CONFIG_USE_UNSLOTH = False

try:
    from config import LOAD_IN_4BIT as CONFIG_LOAD_IN_4BIT
except Exception:
    CONFIG_LOAD_IN_4BIT = False

try:
    from config import LOG_COMPLETION_DETAILS as CONFIG_LOG_COMPLETION_DETAILS
except Exception:
    CONFIG_LOG_COMPLETION_DETAILS = False

from dataset.loader import load_dataset
from prompting.prompt_builder import CountdownPromptBuilder
from rewards.reward_fn import CountdownRewardFunction


@dataclass
class ExperimentPaths:
    output_dir: str
    logs_dir: str
    data_dir: str
    trainer_metrics_csv: str
    trainer_metrics_jsonl: str
    reward_details_jsonl: str
    dataset_snapshot_jsonl: str
    dataset_snapshot_parquet: str
    bucket_counts_csv: str
    experiment_config_json: str


def make_experiment_paths(output_dir: str) -> ExperimentPaths:
    base = Path(output_dir)
    logs_dir = base / "logs"
    data_dir = base / "data_snapshots"
    logs_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    return ExperimentPaths(
        output_dir=str(base),
        logs_dir=str(logs_dir),
        data_dir=str(data_dir),
        trainer_metrics_csv=str(logs_dir / "trainer_metrics.csv"),
        trainer_metrics_jsonl=str(logs_dir / "trainer_metrics.jsonl"),
        reward_details_jsonl=str(logs_dir / "reward_details.jsonl"),
        dataset_snapshot_jsonl=str(data_dir / "train_dataset.jsonl"),
        dataset_snapshot_parquet=str(data_dir / "train_dataset.parquet"),
        bucket_counts_csv=str(data_dir / "bucket_counts.csv"),
        experiment_config_json=str(base / "experiment_config.json"),
    )


def bool_from_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y"}


def get_torch_dtype() -> Any:
    if TORCH_DTYPE == "auto":
        return "auto"
    return getattr(torch, TORCH_DTYPE)


def build_hf_dataset(
    tokenizer,
    dataset_paths: List[str],
    max_samples_per_dataset: Optional[int] = None,
    balance_buckets: bool = True,
    seed: int = 42,
) -> Dataset:
    """Build one HF Dataset from multiple Countdown difficulty parquet files."""
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
                    "target": int(row["target"]),
                    "numbers": list(row["numbers"]),
                    "difficulty_n": int(row["difficulty_n"]),
                    "split": str(row["split"]),
                    "sample_idx": int(row["sample_idx"]),
                }
            )

    rng = random.Random(seed)
    rng.shuffle(processed_rows)
    print(f"Final mixed training dataset size: {len(processed_rows)}")
    return Dataset.from_list(processed_rows)


def save_dataset_artifacts(train_dataset: Dataset, paths: ExperimentPaths) -> None:
    """Save data used for training so plots/tables are reproducible later."""
    train_dataset.to_json(paths.dataset_snapshot_jsonl)
    train_dataset.to_parquet(paths.dataset_snapshot_parquet)

    counts = Counter(train_dataset["difficulty_n"])
    with open(paths.bucket_counts_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["difficulty_n", "count"])
        writer.writeheader()
        for difficulty_n, count in sorted(counts.items()):
            writer.writerow({"difficulty_n": difficulty_n, "count": count})


def _unwrap_completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        if len(completion) > 0 and isinstance(completion[0], dict):
            return completion[0].get("content", "")
        return " ".join(str(x) for x in completion)
    if isinstance(completion, dict):
        return completion.get("content", "")
    return str(completion)


# Global reward logger state. This is intentionally simple and append-only.
_REWARD_LOG_PATH: Optional[str] = None
_REWARD_CALL_ID = 0
_LOG_COMPLETION_DETAILS = True


def configure_reward_logging(path: str, log_completion_details: bool = True) -> None:
    global _REWARD_LOG_PATH, _REWARD_CALL_ID, _LOG_COMPLETION_DETAILS
    _REWARD_LOG_PATH = path
    _REWARD_CALL_ID = 0
    _LOG_COMPLETION_DETAILS = log_completion_details
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("")


def countdown_grpo_reward(
    prompts,
    completions,
    target,
    numbers,
    trainer_state=None,
    difficulty_n=None,
    sample_idx=None,
    split=None,
    **kwargs,
) -> List[float]:
    """Custom Countdown reward function for TRL GRPO."""
    global _REWARD_CALL_ID

    reward_engine = CountdownRewardFunction(tolerance=1e-6, allow_implicit_think_open=True)
    completion_texts = [_unwrap_completion_text(c) for c in completions]

    n_prompts = len(prompts)
    n_completions = len(completion_texts)
    n_targets = len(target)
    n_numbers = len(numbers)

    if not (n_prompts == n_completions == n_targets == n_numbers):
        print("PROMPT EXAMPLE:", repr(prompts[0]) if prompts else "NONE")
        print("COMPLETION EXAMPLE:", repr(completion_texts[0]) if completion_texts else "NONE")
        print("TARGET EXAMPLE:", repr(target[0]) if target else "NONE")
        print("NUMBERS EXAMPLE:", repr(numbers[0]) if numbers else "NONE")
        raise ValueError(
            f"Alignment mismatch: prompts={n_prompts}, completions={n_completions}, "
            f"target={n_targets}, numbers={n_numbers}"
        )

    rewards: List[float] = []
    log_rows: List[Dict[str, Any]] = []

    step = getattr(trainer_state, "global_step", None) if trainer_state is not None else None
    unix_time = time.time()

    for i, completion_text in enumerate(completion_texts):
        result = reward_engine.score_single(
            completion_text=completion_text,
            prompt_numbers=numbers[i],
            target=target[i],
        )
        reward = float(result["reward"])
        rewards.append(reward)

        if _REWARD_LOG_PATH is not None and _LOG_COMPLETION_DETAILS:
            log_rows.append(
                {
                    "reward_call_id": _REWARD_CALL_ID,
                    "global_step": step,
                    "time": unix_time,
                    "reward": reward,
                    "target": int(target[i]) if target[i] is not None else None,
                    "numbers": list(numbers[i]) if numbers[i] is not None else None,
                    "difficulty_n": difficulty_n[i] if difficulty_n is not None else None,
                    "sample_idx": sample_idx[i] if sample_idx is not None else None,
                    "split": split[i] if split is not None else None,
                    "completion_text": completion_text,
                    "reward_result": result,
                }
            )

    if _REWARD_LOG_PATH is not None and log_rows:
        with open(_REWARD_LOG_PATH, "a") as f:
            for row in log_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    _REWARD_CALL_ID += 1
    return rewards


class MetricsToFileCallback(TrainerCallback):
    """Save Trainer logs to CSV and JSONL for plotting later."""

    def __init__(self, csv_path: str, jsonl_path: str):
        self.csv_path = csv_path
        self.jsonl_path = jsonl_path
        self.seen_keys = set()
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        with open(jsonl_path, "w") as f:
            f.write("")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        row = {"step": state.global_step, "epoch": state.epoch, **logs}
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        # CSV field names may grow over time, so rewrite from JSONL each time.
        rows = []
        keys = set()
        with open(self.jsonl_path, "r") as f:
            for line in f:
                item = json.loads(line)
                rows.append(item)
                keys.update(item.keys())
        fieldnames = ["step", "epoch"] + sorted(k for k in keys if k not in {"step", "epoch"})
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


class PatchedGRPOTrainer(GRPOTrainer):
    def _get_train_sampler(self, train_dataset=None):
        return super()._get_train_sampler()


def build_lora_config() -> LoraConfig:
    return LoraConfig(
        r=LORA_CONFIG["r"],
        lora_alpha=LORA_CONFIG["lora_alpha"],
        lora_dropout=LORA_CONFIG["lora_dropout"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )


def load_tokenizer_and_model(hf_token: str):
    """Load tokenizer/model using either Unsloth (FastVisionModel) or standard Transformers.

    Gemma 4 E2B/E4B are natively multimodal (text + image + audio), so the Unsloth
    path uses FastVisionModel, not FastLanguageModel. For Countdown (text-only),
    we disable vision-tower fine-tuning to save VRAM and keep training comparable
    to the standard path.
    """
    use_unsloth = bool_from_env("USE_UNSLOTH", bool(CONFIG_USE_UNSLOTH))
    load_in_4bit = bool_from_env("LOAD_IN_4BIT", bool(CONFIG_LOAD_IN_4BIT))

    if use_unsloth:
        try:
            from unsloth import FastVisionModel, PatchFastRL
            PatchFastRL("GRPO", FastVisionModel)
        except Exception as exc:
            raise ImportError(
                "USE_UNSLOTH=True but Unsloth could not be imported or patched. "
                "Install unsloth in this environment (with transformers==5.5.0 for "
                "Gemma 4 compatibility), then rerun."
            ) from exc

        # Gemma 4 is not yet supported by vLLM; fast_inference=False is required.
        # FastVisionModel auto-detects bf16/fp16 from the GPU, so we don't pass dtype.
        model, processor = FastVisionModel.from_pretrained(
            model_name=MODEL_NAME,
            load_in_4bit=load_in_4bit,
            use_gradient_checkpointing="unsloth",
            fast_inference=False,
            token=hf_token,
        )
        tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        if USE_LORA:
            # Text-only Countdown: skip the vision encoder entirely. This drops
            # ~150M params from LoRA's purview and meaningfully reduces VRAM.
            # The boolean flags replace LORA_CONFIG["target_modules"] in the
            # FastVisionModel API — q/k/v/o/gate/up/down are selected automatically
            # via the attention/MLP toggles below.
            model = FastVisionModel.get_peft_model(
                model,
                finetune_vision_layers=False,
                finetune_language_layers=True,
                finetune_attention_modules=True,
                finetune_mlp_modules=False,
                r=LORA_CONFIG["r"],
                lora_alpha=LORA_CONFIG["lora_alpha"],
                lora_dropout=LORA_CONFIG["lora_dropout"],
                bias="none",
                random_state=42,
                use_rslora=False,
                loftq_config=None,
            )

        # FastVisionModel loads in inference mode by default. Flip it back for training.
        FastVisionModel.for_training(model)
        
        if not hasattr(model, "warnings_issued"):
            model.warnings_issued = {}

        return tokenizer, model, {
            "use_unsloth": True,
            "load_in_4bit": load_in_4bit,
            "loader": "FastVisionModel",
            "finetune_vision_layers": False,
            "finetune_language_layers": True,
        }

    # ---------- Standard Transformers path ----------
    # AutoProcessor is the correct loader for Gemma 4 (it's a multimodal model).
    # We still use AutoModelForCausalLM for the model itself since Countdown
    # never feeds images — keeps the comparison apples-to-apples with unsloth.
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(
        MODEL_NAME, token=hf_token, trust_remote_code=True
    )
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=hf_token,
        trust_remote_code=True,
        torch_dtype=get_torch_dtype(),
        device_map=DEVICE_MAP,
    )

    if USE_LORA:
        model = get_peft_model(model, build_lora_config())
        model.print_trainable_parameters()
        
        
    # Patch: TRL GRPOTrainer expects model.warnings_issued. Gemma4ForConditionalGeneration
    # (the multimodal class AutoModelForCausalLM resolves to) doesn't initialize it in
    # transformers v5, so we add it manually.
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}

    return tokenizer, model, {
        "use_unsloth": False,
        "load_in_4bit": False,
        "loader": "AutoModelForCausalLM",
    }

def main() -> None:
    hf_token = os.environ.get("HF_TOKEN", HF_TOKEN)
    if not hf_token:
        raise ValueError("Set HF_TOKEN in your environment or config.py")

    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
    login(token=hf_token, add_to_git_credential=False)

    output_dir = os.environ.get("OUTPUT_DIR", "outputs/grpo_countdown_mixed_n2_n5_baseline")
    run_name = os.environ.get("RUN_NAME", "grpo_countdown_mixed_n2_n5_baseline")
    paths = make_experiment_paths(output_dir)

    tokenizer, model, load_info = load_tokenizer_and_model(hf_token)

    train_dataset = build_hf_dataset(
        tokenizer=tokenizer,
        dataset_paths=TRAIN_DATASET_PATHS,
        max_samples_per_dataset=None,
        balance_buckets=True,
        seed=42,
    )
    save_dataset_artifacts(train_dataset, paths)
    configure_reward_logging(
        paths.reward_details_jsonl,
        log_completion_details=bool_from_env("LOG_COMPLETION_DETAILS", bool(CONFIG_LOG_COMPLETION_DETAILS)),
    )

    experiment_config = {
        "model_name": MODEL_NAME,
        "train_dataset_paths": TRAIN_DATASET_PATHS,
        "output_dir": output_dir,
        "run_name": run_name,
        "use_lora": USE_LORA,
        "lora_config": LORA_CONFIG,
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "do_sample": DO_SAMPLE,
        "device_map": DEVICE_MAP,
        "torch_dtype": TORCH_DTYPE,
        **load_info,
        "artifacts": asdict(paths),
    }
    with open(paths.experiment_config_json, "w") as f:
        json.dump(experiment_config, f, indent=2)

    training_args = GRPOConfig(
        output_dir=output_dir,
        run_name=run_name,
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
        callbacks=[MetricsToFileCallback(paths.trainer_metrics_csv, paths.trainer_metrics_jsonl)],
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    # Save final trainer state/log history for safety.
    with open(Path(paths.logs_dir) / "trainer_log_history_final.json", "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)


if __name__ == "__main__":
    main()