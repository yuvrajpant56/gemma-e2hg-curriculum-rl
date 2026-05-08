import os
import sys
import json
from typing import Dict, List, Optional, Any

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import (
    MODEL_NAME,
    EVAL_DATASET_PATHS,
    MAX_NEW_TOKENS,
    DEVICE_MAP,
    TORCH_DTYPE,
    HF_TOKEN,
)

from dataset.loader import load_dataset
from prompting.prompt_builder import CountdownPromptBuilder
from inference.gemma_infer import GemmaInference
from rewards.reward_fn import CountdownRewardFunction


# ----------------------------
# Evaluation-time generation settings
# ----------------------------
EVAL_MAX_NEW_TOKENS = MAX_NEW_TOKENS
EVAL_TEMPERATURE = 0.0
EVAL_TOP_P = 1.0
EVAL_DO_SAMPLE = False


def print_separator(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("-" * 100)


def safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def evaluate_one_dataset(
    infer_engine: GemmaInference,
    dataset_path: str,
    max_samples: Optional[int] = 100,
    allow_implicit_think_open: bool = True,
    show_examples: int = 0,
) -> Dict[str, Any]:
    dataset = load_dataset(dataset_path, max_samples=max_samples)
    if not dataset:
        raise ValueError(f"Dataset is empty: {dataset_path}")

    prompt_builder = CountdownPromptBuilder()
    reward_fn = CountdownRewardFunction(
        tolerance=1e-6,
        allow_implicit_think_open=allow_implicit_think_open,
    )

    total_reward = 0.0
    format_valid_count = 0
    answer_tag_count = 0
    exact_correct_count = 0

    syntax_valid_count = 0
    numbers_valid_count = 0
    target_valid_count = 0

    example_logs: List[Dict[str, Any]] = []

    for idx, sample in enumerate(dataset):
        prompt_example = prompt_builder.build_example_for_prompt(sample)
        prompt_data = prompt_builder.generate_prompt(infer_engine.tokenizer, prompt_example)

        generation_result = infer_engine.generate(
            prompt=prompt_data["prompt"],
            max_new_tokens=EVAL_MAX_NEW_TOKENS,
            temperature=EVAL_TEMPERATURE,
            top_p=EVAL_TOP_P,
            do_sample=EVAL_DO_SAMPLE,
        )

        completion = generation_result["completion"]

        score_result = reward_fn.score_single(
            completion_text=completion,
            prompt_numbers=sample["numbers"],
            target=sample["target"],
        )

        checker = score_result["checker_result"]

        reward_value = safe_float(score_result.get("reward", 0.0))
        total_reward += reward_value

        format_valid_count += int(checker.get("format_valid", False))
        answer_tag_count += int(checker.get("has_answer_tag", False))
        exact_correct_count += int(checker.get("is_correct", False))

        syntax_valid_count += int(checker.get("syntax_valid", False))
        numbers_valid_count += int(checker.get("numbers_valid", False))
        target_valid_count += int(checker.get("target_valid", False))

        if idx < show_examples:
            example_logs.append(
                {
                    "idx": idx,
                    "target": sample["target"],
                    "numbers": sample["numbers"],
                    "prompt": prompt_data["prompt"],
                    "completion": completion,
                    "reward": reward_value,
                    "format_valid": checker.get("format_valid"),
                    "has_answer_tag": checker.get("has_answer_tag"),
                    "syntax_valid": checker.get("syntax_valid"),
                    "numbers_valid": checker.get("numbers_valid"),
                    "target_valid": checker.get("target_valid"),
                    "is_correct": checker.get("is_correct"),
                    "expression": checker.get("expression"),
                    "value": checker.get("value"),
                    "error": checker.get("error"),
                }
            )

    n = len(dataset)

    metrics = {
        "num_samples": n,
        "avg_reward": total_reward / n,
        "format_valid_rate": format_valid_count / n,
        "answer_tag_rate": answer_tag_count / n,
        "syntax_valid_rate": syntax_valid_count / n,
        "numbers_valid_rate": numbers_valid_count / n,
        "target_valid_rate": target_valid_count / n,
        "exact_correct_rate": exact_correct_count / n,
        "examples": example_logs,
    }

    return metrics


def evaluate_model_across_buckets(
    model_path: str,
    dataset_paths: Dict[str, str],
    max_samples_per_bucket: Optional[int] = 100,
    allow_implicit_think_open: bool = True,
    show_examples: int = 0,
    tokenizer_name: Optional[str] = None,
    adapter_path: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    infer_engine = GemmaInference(
        model_name=model_path,
        tokenizer_name=tokenizer_name or MODEL_NAME,
        adapter_path=adapter_path,
        hf_token=HF_TOKEN,
        device_map=DEVICE_MAP,
        torch_dtype=TORCH_DTYPE,
    )
    all_results: Dict[str, Dict[str, Any]] = {}

    for bucket_name, dataset_path in dataset_paths.items():
        print_separator(f"EVALUATING {bucket_name} | {dataset_path}")
        metrics = evaluate_one_dataset(
            infer_engine=infer_engine,
            dataset_path=dataset_path,
            max_samples=max_samples_per_bucket,
            allow_implicit_think_open=allow_implicit_think_open,
            show_examples=show_examples,
        )
        all_results[bucket_name] = metrics

        print(f"{bucket_name} | num_samples         : {metrics['num_samples']}")
        print(f"{bucket_name} | avg_reward          : {metrics['avg_reward']:.6f}")
        print(f"{bucket_name} | format_valid_rate   : {metrics['format_valid_rate']:.6f}")
        print(f"{bucket_name} | answer_tag_rate     : {metrics['answer_tag_rate']:.6f}")
        print(f"{bucket_name} | syntax_valid_rate   : {metrics['syntax_valid_rate']:.6f}")
        print(f"{bucket_name} | numbers_valid_rate  : {metrics['numbers_valid_rate']:.6f}")
        print(f"{bucket_name} | target_valid_rate   : {metrics['target_valid_rate']:.6f}")
        print(f"{bucket_name} | exact_correct_rate  : {metrics['exact_correct_rate']:.6f}")

    bucket_keys = list(all_results.keys())

    metric_names = [
        "avg_reward",
        "format_valid_rate",
        "answer_tag_rate",
        "syntax_valid_rate",
        "numbers_valid_rate",
        "target_valid_rate",
        "exact_correct_rate",
    ]

    macro_summary = {}
    for metric_name in metric_names:
        macro_summary[metric_name] = sum(all_results[b][metric_name] for b in bucket_keys) / len(bucket_keys)

    total_samples = sum(all_results[b]["num_samples"] for b in bucket_keys)

    micro_summary = {}
    for metric_name in metric_names:
        weighted_sum = sum(all_results[b][metric_name] * all_results[b]["num_samples"] for b in bucket_keys)
        micro_summary[metric_name] = weighted_sum / total_samples

    summary = {
        "macro_avg": macro_summary,
        "micro_avg": micro_summary,
        "total_samples": total_samples,
    }

    all_results["summary"] = summary

    print_separator("OVERALL SUMMARY | MACRO")
    for key, value in macro_summary.items():
        print(f"{key}: {value:.6f}")

    print_separator("OVERALL SUMMARY | MICRO")
    for key, value in micro_summary.items():
        print(f"{key}: {value:.6f}")

    print(f"total_samples: {total_samples}")

    return all_results


def save_json(data: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def main() -> None:
    # ----------------------------
    # Checkpoint path: non-Unsloth GRPO + LoRA
    # ----------------------------
    output_dir = "outputs/gemma_e2b_grpo_unsloth_no4bit/checkpoint-1600"
    adapter_path = output_dir

    results_dir = os.path.join(output_dir, "evaluation_results")
    os.makedirs(results_dir, exist_ok=True)

    print_separator("EVALUATION CONFIG")
    print(f"Base model      : {MODEL_NAME}")
    print(f"Adapter path    : {adapter_path}")
    print(f"Results dir     : {results_dir}")
    print(f"Max new tokens  : {EVAL_MAX_NEW_TOKENS}")
    print(f"Temperature     : {EVAL_TEMPERATURE}")
    print(f"Top-p           : {EVAL_TOP_P}")
    print(f"Do sample       : {EVAL_DO_SAMPLE}")

    # ----------------------------
    # 1. Check whether LoRA adapter exists
    # ----------------------------
    adapter_config = os.path.join(adapter_path, "adapter_config.json")
    adapter_weights = os.path.join(adapter_path, "adapter_model.safetensors")

    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"Checkpoint directory not found: {adapter_path}")

    if not os.path.exists(adapter_config):
        raise FileNotFoundError(f"Missing LoRA adapter config: {adapter_config}")

    if not os.path.exists(adapter_weights):
        raise FileNotFoundError(f"Missing LoRA adapter weights: {adapter_weights}")

    # ----------------------------
    # 2. Base model evaluation
    # ----------------------------
    print_separator("BASE MODEL EVALUATION")
    base_results = evaluate_model_across_buckets(
        model_path=MODEL_NAME,
        tokenizer_name=MODEL_NAME,
        dataset_paths=EVAL_DATASET_PATHS,
        max_samples_per_bucket=100,
        allow_implicit_think_open=True,
        show_examples=1,
        adapter_path=None,
    )

    save_json(base_results, os.path.join(results_dir, "eval_base.json"))

    # ----------------------------
    # 3. LoRA checkpoint evaluation
    # ----------------------------
    print_separator("LORA CHECKPOINT EVALUATION | WITH UNSLOTH")
    lora_results = evaluate_model_across_buckets(
        model_path=MODEL_NAME,
        tokenizer_name=MODEL_NAME,
        dataset_paths=EVAL_DATASET_PATHS,
        max_samples_per_bucket=100,
        allow_implicit_think_open=True,
        show_examples=1,
        adapter_path=adapter_path,
    )

    save_json(lora_results, os.path.join(results_dir, "eval_lora_with_unsloth_no4bit.json"))

    print_separator("DONE")
    print(f"Saved base results to: {os.path.join(results_dir, 'eval_base.json')}")
    print(f"Saved LoRA results to: {os.path.join(results_dir, 'eval_lora_with_unsloth.json')}")

if __name__ == "__main__":
    main()