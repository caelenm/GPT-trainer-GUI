from GPT2_trainer import modelAndDatasetPicker, tokenize, train
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import DataCollatorForLanguageModeling
import torch
import gc
import argparse


# Main function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt2')
    parser.add_argument('--dataset', type=str, default='flights')
    parser.add_argument('--max_steps', type=int, default=250)
    parser.add_argument('--k-folds', type=int, default=2)
    args = parser.parse_args()

    # Free memory
    torch.cuda.empty_cache()
    gc.collect()    

    model_name = args.model
    config_name = args.dataset
    max_steps = args.max_steps
    kfolds = args.k_folds
    dataset_name="taskmaster2"
    
    if config_name == "food-ordering":
        subset_size = 1000
    if config_name == "music":
        subset_size = 1500
    else:
        subset_size=2000
    model_name, dataset = modelAndDatasetPicker(model_name, dataset_name, config_name, subset_size)
    # Load the tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    # Tokenize the dataset
    tokenized_dataset = tokenize(tokenizer, dataset)
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  
    )

    train(model, tokenized_dataset, data_collator, k=kfolds, max_steps=max_steps)

if __name__ == "__main__":
    main()
