from transformers import AutoProcessor, AutoModelForCausalLM
import torch

proc = AutoProcessor.from_pretrained("google/gemma-4-E2B-it")
tok = proc.tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-4-E2B-it", dtype=torch.bfloat16, device_map="auto"
)

user_text = "Using the numbers [51, 54], create an equation that equals 3."
messages = [{"role": "user", "content": user_text}]

# Test 1: Proper chat template, with BOS
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print("=" * 60)
print("PROMPT WITH CHAT TEMPLATE (raw string):")
print(repr(prompt))
print("=" * 60)

ids = tok(prompt, return_tensors="pt", add_special_tokens=True).input_ids.cuda()
out = model.generate(ids, max_new_tokens=150, do_sample=False)
print("\nTEST 1 — chat template + BOS:")
print(tok.decode(out[0][ids.shape[1]:]))

# Test 2: Chat template, BOS stripped (mimics what TRL does)
ids2 = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
out2 = model.generate(ids2, max_new_tokens=150, do_sample=False)
print("\nTEST 2 — chat template, NO BOS (this is what TRL does):")
print(tok.decode(out2[0][ids2.shape[1]:]))

# Test 3: Raw text only (no chat template) — what your prompt builder might be doing
raw = user_text + "\n<think>"
ids3 = tok(raw, return_tensors="pt", add_special_tokens=True).input_ids.cuda()
out3 = model.generate(ids3, max_new_tokens=150, do_sample=False)
print("\nTEST 3 — raw text + <think>, no chat template:")
print(tok.decode(out3[0][ids3.shape[1]:]))


import sys; sys.path.insert(0, "/mmfs1/scratch/jacks.local/pkhanal2568/Gemma_Hackathon/Grpo_Training")
from prompting.prompt_builder import CountdownPromptBuilder

pb = CountdownPromptBuilder()
ex = pb.build_example_for_prompt({"target": 3, "numbers": [51,54], "difficulty_n": 2, "split": "train", "sample_idx": 0})
out = pb.generate_prompt(tok, ex)
print("PROMPT:")
print(repr(out["prompt"]))
print()

ids = tok(out["prompt"], return_tensors="pt", add_special_tokens=False).input_ids.cuda()
gen = model.generate(ids, max_new_tokens=200, do_sample=False)
print("COMPLETION:")
print(tok.decode(gen[0][ids.shape[1]:]))