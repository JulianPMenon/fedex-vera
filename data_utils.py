import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    AdamW,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm
import numpy as np
import pandas as pd
from peft import get_peft_model, LoraConfig, TaskType, VeraConfig


def load_and_preprocess_data(task):

    if "mnli" in task:
        dataset = load_dataset("data/glue", "mnli")
    else:
        dataset = load_dataset("data/glue", task)

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    def tokenize_function(examples):

        # Handle different input formats
        if "premise" in examples and "hypothesis" in examples:
            # MNLI and similar tasks
            return tokenizer(
                examples["premise"],
                examples["hypothesis"],
                truncation=True,
                padding="max_length",
                max_length=128,
            )
        elif "question" in examples and "sentence" in examples:
            # QNLI and similar tasks
            return tokenizer(
                examples["question"],
                examples["sentence"],
                truncation=True,
                padding="max_length",
                max_length=128,
            )
        elif "sentence1" in examples and "sentence2" in examples:
            # MRPC, STS-B
            return tokenizer(
                examples["sentence1"],
                examples["sentence2"],
                truncation=True,
                padding="max_length",
                max_length=128,
            )
        elif "question1" in examples and "question2" in examples:
            # QQP
            return tokenizer(
                examples["question1"],
                examples["question2"],
                truncation=True,
                padding="max_length",
                max_length=128,
            )
        elif "sentence" in examples:
            # CoLA, SST-2
            return tokenizer(
                examples["sentence"],
                truncation=True,
                padding="max_length",
                max_length=128,
            )
        else:
            raise ValueError(f"Unexpected format for task {task}")

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    if task == "cola":
        tokenized_datasets = tokenized_datasets.remove_columns(["sentence", "idx"])
    elif task == "sst2":
        tokenized_datasets = tokenized_datasets.remove_columns(["sentence", "idx"])
    elif task == "mrpc":
        tokenized_datasets = tokenized_datasets.remove_columns(
            ["sentence1", "sentence2", "idx"]
        )
    elif task == "qqp":
        tokenized_datasets = tokenized_datasets.remove_columns(
            ["question1", "question2", "idx"]
        )
    elif task == "stsb":
        tokenized_datasets = tokenized_datasets.remove_columns(
            ["sentence1", "sentence2", "idx"]
        )
    elif task == "qnli":
        tokenized_datasets = tokenized_datasets.remove_columns(
            ["question", "sentence", "idx"]
        )
    elif task == "rte":
        tokenized_datasets = tokenized_datasets.remove_columns(
            ["sentence1", "sentence2", "idx"]
        )
    elif task == "wnli":
        tokenized_datasets = tokenized_datasets.remove_columns(
            ["sentence1", "sentence2", "idx"]
        )
    elif task == "mnli_matched" or task == "mnli_mismatched" or task == "mnli":
        tokenized_datasets = tokenized_datasets.remove_columns(
            ["premise", "hypothesis", "idx"]
        )
    else:
        raise ValueError(f"Unexpected task {task}")

    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    if (
        task == "cola"
        or task == "sst2"
        or task == "mrpc"
        or task == "qqp"
        or task == "stsb"
        or task == "qnli"
        or task == "rte"
        or task == "wnli"
    ):
        train_dataset = tokenized_datasets["train"]
        val_dataset = tokenized_datasets["validation"]
        test_dataset = tokenized_datasets["test"]
    elif task == "mnli_matched":
        train_dataset = tokenized_datasets["train"]
        val_dataset = tokenized_datasets["validation_matched"]
        test_dataset = tokenized_datasets["test_matched"]
    elif task == "mnli_mismatched":
        train_dataset = tokenized_datasets["train"]
        val_dataset = tokenized_datasets["validation_mismatched"]
        test_dataset = tokenized_datasets["test_mismatched"]

    return train_dataset, val_dataset, test_dataset


def create_dataloader(dataset, args):
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=False)


def create_client_dataloaders_nlg(dataset, args):
    client_data = [[] for _ in range(args.num_clients)]
    for data in dataset:
        client_idx = np.random.randint(args.num_clients)
        client_data[client_idx].append(data)
    return client_data


def create_client_dataloaders(dataset, args):
    client_data = [[] for _ in range(args.num_clients)]
    for data in dataset:
        client_idx = np.random.randint(args.num_clients)
        client_data[client_idx].append(data)
    return [
        DataLoader(cd, batch_size=args.batch_size, shuffle=True) for cd in client_data
    ]


def create_e2e_data():
    def preprocess_function(examples):
        inputs = examples["meaning_representation"]
        targets = examples["human_reference"]

        # Combine the input-output pair into a single text
        model_inputs = [
            f"{input_} -> {target} <|endoftext|>"
            for input_, target in zip(inputs, targets)
        ]
        only_inputs = [f"{input_} ->" for input_, target in zip(inputs, targets)]

        # Tokenize the combined inputs
        tokenized_inputs = tokenizer(
            model_inputs,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        tokenized_only_inputs = tokenizer(
            only_inputs,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Labels are the same as input_ids but shift them for next-token prediction
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].clone()

        # Set the labels to -100 where attention mask is 0 (this will ignore padding in loss computation)
        tokenized_inputs["labels"][tokenized_inputs["attention_mask"] == 0] = -100
        # set the labels to -100 where meaning representation input ids are present
        tokenized_inputs["labels"][tokenized_only_inputs["attention_mask"] == 1] = -100

        return tokenized_inputs

    dataset = load_dataset("tuetschek/e2e_nlg")
    from transformers import GPT2Tokenizer

    # Load the GPT-2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = (
        tokenizer.eos_token
    )  # GPT-2 doesn't have a pad token, so we set it to the eos token
    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    return (
        tokenized_datasets["train"],
        tokenized_datasets["validation"],
        tokenized_datasets["test"],
        tokenizer,
    )
