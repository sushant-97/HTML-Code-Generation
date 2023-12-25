# Import libraries
import argparse
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import get_cosine_schedule_with_warmup
from transformers import GenerationConfig
from pathlib import Path

import bitsandbytes as bnb
import wandb
from types import SimpleNamespace
from tqdm.auto import tqdm

from data_preprocess import DatasetPreprocessor
from model import ModelManager
from eval import Accuracy, compute_rouge # Assuming you have an 'Accuracy' class or function in eval.py

# Define the ModelManager class in model.py
# Define the DatasetPreprocessor class in data_preprocess.py

# Define Configuration and other Utilities
def setup_config():
    config = SimpleNamespace(
        lr=2e-4,
        n_eval_samples=10,
        max_seq_len=1024,
        epochs=3,
        gradient_accumulation_steps=4,
        batch_size=4,
        log_model=True,
        gradient_checkpointing=True,
        freeze_embed=True,
        seed=42,
    )
    return config

# Define Loss and Other Training Functions
def loss_fn(x, y):
    return torch.nn.functional.cross_entropy(x.view(-1, x.shape[-1]), y.view(-1))

def to_gpu(tensor_dict):
    return {k: v.to('cuda') for k, v in tensor_dict.items()}

# Define Validation Function
@torch.no_grad()
def validate(model, eval_dataloader):
    model.eval();
    eval_acc = Accuracy()
    loss, total_steps = 0., 0
    for step, batch in enumerate(pbar:=tqdm(eval_dataloader, leave=False)):
        pbar.set_description(f"doing validation")
        batch = to_gpu(batch)
        total_steps += 1
        with torch.amp.autocast("cuda", dtype=torch.float16):
            out = model(**batch)
            loss += loss_fn(out.logits, batch["labels"])  # you could use out.loss and not shift the dataset
        eval_acc.update(out.logits, batch["labels"])
    # we log results at the end
    wandb.log({"eval/loss": loss.item() / total_steps,
               "eval/accuracy": eval_acc.compute()})

# Define Training Function
def train_model(model, train_dataloader, eval_dataloader, optim, scheduler, config):
    acc = Accuracy()
    model.train()
    train_step = 0
    for epoch in tqdm(range(config.epochs)):
        for step, batch in enumerate(tqdm(train_dataloader)):
            batch = to_gpu(batch)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = model(**batch)
                loss = loss_fn(out.logits, batch["labels"]) / config.gradient_accumulation_steps  # you could use out.loss and not shift the dataset  
                loss.backward()
            if step%config.gradient_accumulation_steps == 0:
                # we can log the metrics to W&B
                wandb.log({"train/loss": loss.item() * config.gradient_accumulation_steps,
                        "train/accuracy": acc.update(out.logits, batch["labels"]),
                        "train/learning_rate": scheduler.get_last_lr()[0],
                        "train/global_step": train_step})
                optim.step()
                scheduler.step()
                optim.zero_grad(set_to_none=True)
                train_step += 1
        validate(model, eval_dataloader)
        model.train()


def generate(prompt, model_id, model, tokenizer):
    gen_config = GenerationConfig.from_pretrained(model_id)
    test_config = SimpleNamespace(
        max_new_tokens=256,
        gen_config=gen_config)

    tokenized_prompt = tokenizer(prompt, return_tensors='pt')['input_ids'].cuda()
    with torch.inference_mode():
        output = model.generate(tokenized_prompt, 
                            max_new_tokens=test_config.max_new_tokens, 
                            generation_config=test_config.gen_config)
        
    return tokenizer.decode(output[0][len(tokenized_prompt[0]):], skip_special_tokens=True)

def predictions(examples, model_id, model, tokenizer, test_config, log=False, table_name="predictions"):
    table = wandb.Table(columns=["prompt", "generation", "concat", "output", "max_new_tokens", "temperature", "top_p"])
    for example in tqdm(examples, leave=False):
        prompt, label = example["prompt"], example["output"]
        prediction  = generate(prompt, model_id, model, tokenizer)
        table.add_data(prompt, prediction, prompt+prediction, label, test_config.max_new_tokens, test_config.gen_config.temperature, test_config.gen_config.top_p)
    if log:
        wandb.log({table_name:table})
    return prediction, label

# Save Model Function
def save_model(model, model_name, models_folder="models", log=False):
    """Save the model to wandb as an artifact
    Args:
        model (nn.Module): Model to save.
        model_name (str): Name of the model.
        models_folder (str, optional): Folder to save the model. Defaults to "models".
    """
    model_name = f"{wandb.run.id}_{model_name}"
    file_name = Path(f"{models_folder}/{model_name}")
    file_name.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(file_name, safe_serialization=True)
    # save tokenizer for easy inference
    tokenizer = AutoTokenizer.from_pretrained(model.name_or_path)
    tokenizer.save_pretrained(model_name)
    if log:
        at = wandb.Artifact(model_name, type="model")
        at.add_dir(file_name)
        wandb.log_artifact(at)

# Main Function
def main():
    # Setup Configuration and Seed
    config = setup_config()
    torch.manual_seed(config.seed)

    # Prepare Data
    preprocessor = DatasetPreprocessor()
    dataset_id = 'jawerty/html_dataset'
    train_dataloader, eval_dataloader = preprocessor.create_data_loaders(dataset_id, batch_size=config.batch_size)

    # Initialize ModelManager
    model_id = "meta-llama/Llama-2-7b-hf"
    model_mgr = ModelManager(model_name=model_id)
    model = model_mgr.model
    tokenizer = model_mgr.tokenizer

    # Setup Optimizer and Scheduler
    optim = bnb.optim.Adam8bit(model.parameters(), lr=config.lr, betas=(0.9, 0.99), eps=1e-5)
    scheduler = get_cosine_schedule_with_warmup(
        optim,
        num_training_steps=config.total_train_steps,
        num_warmup_steps=config.total_train_steps // 10,
    )

    wandb.init(project="html-generation", # the project I am working on
            tags=["baseline","7b"],
            job_type="train",
            config=config) # the Hyperparameters I want to keep track of

    # Training Loop
    train_model(model, train_dataloader, eval_dataloader, optim, scheduler, config)

    # Testing.. Ideally, this should be in a separate data but
    # because of only 43 data points, we are using the eval dataset
    # Eval Dataset evaluation
    validate(model, eval_dataloader)

    # Merge Model
    model = model.merge_and_unload()

    with wandb.init(project="html-generation", # the project I am working on
           job_type="eval",
           config=config): # the Hyperparameters I want to keep track of

        model.eval()
        gen_config = GenerationConfig.from_pretrained(model_id)
        test_config = SimpleNamespace(
            max_new_tokens=256,
            gen_config=gen_config)
            
        predictions, references = predictions(eval_dataloader, model_id, model, tokenizer, test_config, log=True, table_name="eval_predictions")
        aggregated_scores = compute_rouge(predictions=predictions, references=references)
        
        wandb.log({'val_rouge1': aggregated_scores['rouge1'],
                'val_rouge2': aggregated_scores['rouge2'],
                'val_rougeL': aggregated_scores['rougeL']})

    # Compute Rouge Score

    # Save the final model
    save_model(model, model_name="final_merged_checkpoint")

    # Push to Hub
    model.push_to_hub("llama2-fine-tuned-jawerty_html_dataset")
    tokenizer.push_to_hub("llama2-fine-tuned-jawerty_html_dataset")

    # # we save the model checkpoint at WANDB
    # save_model(model, model_name= model_name.replace("/", "_"), models_folder="models/", log=config.log_model)  
    # wandb.finish()

if __name__ == '__main__':
    main()
