import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from safetensors.torch import load_file
import os
import json

# Find the latest best_model directory
model_dirs = [d for d in os.listdir() if d.startswith("best_model")]
if not model_dirs:
    raise FileNotFoundError("No best_model directory found!")

latest_model_dir = sorted(model_dirs)[-1]  # Get the most recent one
model_path = os.path.join(latest_model_dir, "model.safetensors")
config_path = os.path.join(latest_model_dir, "config.json")

with open(config_path, "r") as f:
    config = json.load(f)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
state_dict = load_file(model_path)

# Init model and load weights
model = GPT2LMHeadModel.from_pretrained("gpt2", state_dict=state_dict)
model.eval()

def generate_text(prompt, max_length=50, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_k=50,
            top_p=0.95,
            no_repeat_ngram_size=2,
            do_sample=True
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


prompt = input("Enter a prompt: ")
generated_text = generate_text(prompt)
print("\nGenerated Text:\n", generated_text)
