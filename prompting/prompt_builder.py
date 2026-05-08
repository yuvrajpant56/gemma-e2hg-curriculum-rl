from typing import Dict, List, Any

class CountdownPromptBuilder:
    """
    Responsible only for building prompts and optional reasoning traces.
    This separation will help later when you move to GRPO or reward functions.
    """

    def _construct_reasoning_trace(self, reasoning_steps: List[str]) -> List[str]:
        """Construct reasoning trace from reasoning steps."""
        if not reasoning_steps:
            return []

        reasoning_trace = []
        n_r = len(reasoning_steps) - 1
        for i, step in enumerate(reasoning_steps):
            if 0 < i < n_r:
                reasoning_trace.append(f"Step {i}: {step}")
        reasoning_trace.append(f"Final Result: {reasoning_steps[-1]}")
        return reasoning_trace

    def build_example_for_prompt(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert your raw dataset row into a structure that mirrors the style
        of your future RL/reward-model pipeline.
        """
        return {
            "reward_model": {
                "ground_truth": {
                    "target": example["target"],
                    "numbers": example["numbers"],
                }
            },
            "reasoning_steps": example.get("reasoning_steps", []),
            "meta": {
                "difficulty_n": example.get("difficulty_n"),
                "split": example.get("split"),
                "sample_idx": example.get("sample_idx"),
            },
        }

    def generate_prompt(self, tokenizer, example: Dict[str, Any]) -> Dict[str, Any]:
        data = example.get("reward_model", {}).get("ground_truth", {})
        target = data.get("target")
        numbers = data.get("numbers")

        reasoning_steps = example.get("reasoning_steps", [])
        reasoning_trace = self._construct_reasoning_trace(reasoning_steps)

        system_msg = (
            "You are a Countdown puzzle solver. Given a list of numbers and a "
            "target, construct an arithmetic expression that evaluates exactly "
            "to the target.\n\n"
            "Rules (must obey exactly):\n"
            "- Use only the provided numbers.\n"
            "- Use each number exactly once.\n"
            "- Use only +, -, *, /, and parentheses.\n"
            "- Do not introduce any new numbers.\n"
            "- The final equation must evaluate exactly to the target.\n\n"
            "Output format (must match exactly):\n"
            "<think>your reasoning here</think>\n"
            "<answer>your final equation here</answer>"
        )

        user_msg = (
            f"Using the numbers {numbers}, create an equation that equals {target}."
        )

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,   # appends <|turn>model\n so the model knows to generate
        )

        return {
            "prompt": prompt,
            "target": target,
            "numbers": numbers,
            "reasoning_trace": reasoning_trace,
            "meta": example.get("meta", {}),
        }