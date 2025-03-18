from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
import datetime
import torch
from sklearn.model_selection import KFold
import gc


#CUDA check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


torch.backends.cudnn.benchmark = True
current_time = datetime.datetime.now()
formatted_time = current_time.strftime("%Y%m%d_%H%M%S")

# load dataset and model, and limit to a subset of training examples.
def modelAndDatasetPicker(model_name, dataset_name, config_name, subset_size):
    dataset = load_dataset(dataset_name, config_name, trust_remote_code=True)
    dataset["train"] = dataset["train"].select(range(subset_size))
    return model_name, dataset


# Take dataset and tokenize/process it for training
def tokenize(tokenizer, dataset):
    # Set padding token to the EOS token for GPT-2.
    tokenizer.pad_token = tokenizer.eos_token
    max_length = 142  

    def preprocess_function(examples):
        all_tokenized = {"input_ids": [], "attention_mask": [], "labels": []}
        for utterance_list in examples["utterances"]:
            for utterance in utterance_list:
                if utterance["speaker"].upper() != "ASSISTANT": #ignores non 'ASSISTANT" text since gpt2 is not meant for QA training
                    continue
                tokenized = tokenizer(
                    utterance["text"],
                    truncation=True,
                    padding="max_length",
                    max_length=max_length
                )
                all_tokenized["input_ids"].append(tokenized["input_ids"])
                all_tokenized["attention_mask"].append(tokenized["attention_mask"])
                all_tokenized["labels"].append(tokenized["input_ids"].copy())
        return all_tokenized

    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names  
    )
    
    print("Sample tokenized output:", tokenized_dataset["train"][0])
    return tokenized_dataset


# train the model using K-Fold Cross Validation.
def train(model, dataset, data_collator, k, max_steps=300):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

   
    training_args = TrainingArguments(
        output_dir=f"./results_{formatted_time}",
        dataloader_num_workers=12,
        fp16=True,
        gradient_accumulation_steps=4, 
        per_device_train_batch_size=16, 
        num_train_epochs=3,
        max_steps=max_steps,
        logging_dir='./logs',
        logging_steps=100,
        lr_scheduler_type="linear",
        eval_strategy="epoch",  
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",  
        greater_is_better=False,
        save_strategy="epoch",  
        save_total_limit=1,   
    )

    best_model = None
    best_loss = float("inf")

    #k folds
 
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset['train'])):
        print(f"Training fold {fold + 1}/{k}")
        
        train_fold = dataset['train'].select(train_idx)
        val_fold = dataset['train'].select(val_idx)
        

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_fold,
            eval_dataset=val_fold,
            data_collator=data_collator,
        )
       
        trainer.train()

        eval_results = trainer.evaluate(eval_dataset=val_fold)
        eval_loss = eval_results["eval_loss"]
        print(f"Fold {fold + 1} evaluation loss: {eval_loss}")
        if eval_loss < best_loss:
            best_loss = eval_loss
            best_model = model
            # Save the best model.
            best_model.save_pretrained(f"./best_model_{formatted_time}")
            print(f"New best model saved with eval_loss {best_loss}")
        
        #free memory
        torch.cuda.empty_cache()
        gc.collect()

    return best_model
