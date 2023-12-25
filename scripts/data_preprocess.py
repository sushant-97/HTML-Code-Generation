import os
from datasets import load_dataset
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling, AutoTokenizer

class DatasetPreprocessor:
    def __init__(self, model_id='meta-llama/Llama-2-7b-hf', max_sequence_len=1024):
        self.seed = 42
        self.max_sequence_len = max_sequence_len
        torch.manual_seed(self.seed)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def create_data_loaders(self, dataset_id, batch_size=4):
        dataset = load_dataset(dataset_id, split="train")
        train_dataset, eval_dataset = train_test_split(dataset, test_size=0.2)

        train_prompts = [self.create_html_prompt(row) for row in train_dataset["label"]]
        eval_prompts = [self.create_html_prompt(row) for row in eval_dataset["label"]]

        train_outputs = self.pad_eos(train_dataset["html"])
        eval_outputs = self.pad_eos(eval_dataset["html"])

        train_dataset = [{"prompt": s, "output": t, "example": s + t} for s, t in zip(train_prompts, train_outputs)]
        eval_dataset = [{"prompt": s, "output": t, "example": s + t} for s, t in zip(eval_prompts, eval_outputs)]

        train_ds_packed = self.pack(train_dataset)
        eval_ds_packed = self.pack(eval_dataset)

        return self.data_loader(train_ds_packed, eval_ds_packed, batch_size)

    def pack(self, dataset):
        tkds_ids = self.tokenizer([s["example"] for s in dataset])["input_ids"]
        all_token_ids = []
        for tokenized_input in tkds_ids:
            all_token_ids.extend(tokenized_input)

        packed_ds = []
        for i in range(0, len(all_token_ids), self.max_sequence_len+1):
            input_ids = all_token_ids[i : i + self.max_sequence_len+1]
            if len(input_ids) == (self.max_sequence_len+1):
                packed_ds.append({"input_ids": input_ids[:-1], "labels": input_ids[1:]})
        return packed_ds

    @staticmethod
    def create_html_prompt(row):
        return ("Below is an instruction that describes a task. "
                "You will be provided a prompt and based on it you have to generate HTML code. Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{}\n\n### Response:\n").format(row)

    @staticmethod
    def pad_eos(ds):
        EOS_TOKEN = "</s>"
        return [f"{row}{EOS_TOKEN}" for row in ds]

    def data_loader(self, train_ds_packed, eval_ds_packed, batch_size=32):
        collate_fn = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        train_dataloader = DataLoader(train_ds_packed, batch_size=batch_size, collate_fn=collate_fn)
        eval_dataloader = DataLoader(eval_ds_packed, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
        return train_dataloader, eval_dataloader
