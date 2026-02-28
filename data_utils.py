import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, RobertaTokenizer


_TASK_REMOVE_COLUMNS = {
    "cola": ["sentence", "idx"],
    "sst2": ["sentence", "idx"],
    "mrpc": ["sentence1", "sentence2", "idx"],
    "qqp": ["question1", "question2", "idx"],
    "stsb": ["sentence1", "sentence2", "idx"],
    "qnli": ["question", "sentence", "idx"],
    "rte": ["sentence1", "sentence2", "idx"],
    "wnli": ["sentence1", "sentence2", "idx"],
    "mnli": ["premise", "hypothesis", "idx"],
    "mnli_matched": ["premise", "hypothesis", "idx"],
    "mnli_mismatched": ["premise", "hypothesis", "idx"],
}

_TASK_SPLITS = {
    "cola": ("train", "validation", "test"),
    "sst2": ("train", "validation", "test"),
    "mrpc": ("train", "validation", "test"),
    "qqp": ("train", "validation", "test"),
    "stsb": ("train", "validation", "test"),
    "qnli": ("train", "validation", "test"),
    "rte": ("train", "validation", "test"),
    "wnli": ("train", "validation", "test"),
    "mnli": ("train", "validation_matched", "test_matched"),
    "mnli_matched": ("train", "validation_matched", "test_matched"),
    "mnli_mismatched": ("train", "validation_mismatched", "test_mismatched"),
}


def load_and_preprocess_data(task):
    dataset = load_dataset("data/glue", "mnli" if "mnli" in task else task)

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    def tokenize_function(examples):
        # Route by field names to avoid task-specific branching elsewhere.
        if "premise" in examples and "hypothesis" in examples:
            return tokenizer(
                examples["premise"],
                examples["hypothesis"],
                truncation=True,
                padding="max_length",
                max_length=128,
            )
        if "question" in examples and "sentence" in examples:
            return tokenizer(
                examples["question"],
                examples["sentence"],
                truncation=True,
                padding="max_length",
                max_length=128,
            )
        if "sentence1" in examples and "sentence2" in examples:
            return tokenizer(
                examples["sentence1"],
                examples["sentence2"],
                truncation=True,
                padding="max_length",
                max_length=128,
            )
        if "question1" in examples and "question2" in examples:
            return tokenizer(
                examples["question1"],
                examples["question2"],
                truncation=True,
                padding="max_length",
                max_length=128,
            )
        if "sentence" in examples:
            return tokenizer(
                examples["sentence"],
                truncation=True,
                padding="max_length",
                max_length=128,
            )
        raise ValueError(f"Unexpected format for task {task}")

    # Normalize task-specific schemas to a single tokenized format.
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    if task not in _TASK_REMOVE_COLUMNS:
        raise ValueError(f"Unexpected task {task}")

    tokenized_datasets = tokenized_datasets.remove_columns(
        _TASK_REMOVE_COLUMNS[task]
    )
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    train_key, val_key, test_key = _TASK_SPLITS[task]
    train_dataset = tokenized_datasets[train_key]
    val_dataset = tokenized_datasets[val_key]
    test_dataset = tokenized_datasets[test_key]

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
    client_data = create_client_datasets(dataset, args)
    return [
        DataLoader(cd, batch_size=args.batch_size, shuffle=True) for cd in client_data
    ]


def create_client_datasets(dataset, args):
    client_data = [[] for _ in range(args.num_clients)]
    for data in dataset:
        client_idx = np.random.randint(args.num_clients)
        client_data[client_idx].append(data)
    return client_data


def create_e2e_data():
    def preprocess_function(examples):
        inputs = examples["meaning_representation"]
        targets = examples["human_reference"]

        model_inputs = [
            f"{input_} -> {target} <|endoftext|>"
            for input_, target in zip(inputs, targets)
        ]
        only_inputs = [f"{input_} ->" for input_, target in zip(inputs, targets)]

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

        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].clone()
        # Avoid loss on padding and prompt tokens to focus updates on targets.
        tokenized_inputs["labels"][tokenized_inputs["attention_mask"] == 0] = -100
        tokenized_inputs["labels"][tokenized_only_inputs["attention_mask"] == 1] = -100

        return tokenized_inputs

    dataset = load_dataset("tuetschek/e2e_nlg")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    return (
        tokenized_datasets["train"],
        tokenized_datasets["validation"],
        tokenized_datasets["test"],
        tokenizer,
    )
