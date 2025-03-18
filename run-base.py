import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Add a padding token to the tokenizer
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Update the model to resize token embeddings to accommodate the new padding token
model.resize_token_embeddings(len(tokenizer))

# Function to generate text
def generate_text(prompt, max_length=100, max_new_tokens=50):
    inputs = tokenizer.encode_plus(prompt, return_tensors="pt", padding="max_length", max_length=max_length, truncation=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Generate text
    outputs = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

# Example usage
if __name__ == "__main__":
    prompt = input("Enter a prompt for base: ")
    generated_text = generate_text(prompt)
    print(generated_text)
