# inference/gemma_infer.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Any, Optional
from peft import PeftModel


class GemmaInference:
    def __init__(
        self,
        model_name: str,
        hf_token: Optional[str] = None,
        adapter_path: Optional[str] = None,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        tokenizer_name: Optional[str] = None,
    ):
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name or model_name
        self.adapter_path = adapter_path
        self.hf_token = hf_token

        print(f"[INFO] Loading tokenizer from: {self.tokenizer_name}", flush=True)
        print(f"[INFO] Loading base model from: {self.model_name}", flush=True)
        print(f"[INFO] Adapter path: {self.adapter_path}", flush=True)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name,
            token=self.hf_token,
            trust_remote_code=True,
        )

        # Gemma models often do not have a separate pad token.
        # For generation, using eos_token as pad_token is standard.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Important for decoder-only batched generation.
        self.tokenizer.padding_side = "left"

        dtype = self._resolve_dtype(torch_dtype)

        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            token=self.hf_token,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map=device_map,
        )

        if self.adapter_path is not None:
            print(f"[INFO] Loading LoRA adapter from: {self.adapter_path}", flush=True)
            self.model = PeftModel.from_pretrained(
                base_model,
                self.adapter_path,
            )
        else:
            self.model = base_model

        self.model.eval()

        print("[INFO] Model loaded successfully.", flush=True)

    def _resolve_dtype(self, torch_dtype: str):
        if torch_dtype == "auto":
            if torch.cuda.is_available():
                if torch.cuda.is_bf16_supported():
                    return torch.bfloat16
                return torch.float16
            return torch.float32

        if torch_dtype in ["bfloat16", "bf16"]:
            return torch.bfloat16

        if torch_dtype in ["float16", "fp16"]:
            return torch.float16

        if torch_dtype in ["float32", "fp32"]:
            return torch.float32

        raise ValueError(f"Unsupported torch_dtype: {torch_dtype}")

    def _get_input_device(self) -> torch.device:
        return next(self.model.parameters()).device

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 1.0,
        do_sample: bool = False,
    ) -> Dict[str, Any]:

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
            truncation=False,
        )

        input_device = self._get_input_device()
        inputs = {k: v.to(input_device) for k, v in inputs.items()}

        generation_kwargs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
        }

        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p

        with torch.no_grad():
            output_ids = self.model.generate(**generation_kwargs)

        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0][prompt_len:]

        completion = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        full_text = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
        )

        return {
            "completion": completion,
            "full_text": full_text,
        }