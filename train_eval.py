import os

import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from transformers import Trainer, TrainingArguments, get_linear_schedule_with_warmup

from data_utils import create_dataloader


def train_client(model, dataloader, args, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(dataloader) * args.local_epochs
    num_warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps
    )

    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)
    model.train()
    for _epoch in range(args.local_epochs):
        for _step, data in enumerate(tqdm(dataloader)):
            data = {k: v.to(device) for k, v in data.items()}

            with autocast(enabled=use_amp):
                outputs = model(**data)
                loss = outputs.loss

            # AMP keeps GPU memory stable on long sequences.
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

    return model.state_dict()


def calculate_metrics(all_true_labels, all_predictions, task):
    if task == "cola":
        return accuracy_score(all_true_labels, all_predictions), matthews_corrcoef(
            all_true_labels, all_predictions
        )
    if task in ["sst2", "qnli", "rte", "wnli"]:
        return accuracy_score(all_true_labels, all_predictions), None
    if task == "mrpc":
        return f1_score(all_true_labels, all_predictions), accuracy_score(
            all_true_labels, all_predictions
        )
    if task == "stsb":
        return (
            pearsonr(all_true_labels, all_predictions)[0],
            spearmanr(all_true_labels, all_predictions)[0],
        )
    if task == "qqp":
        return accuracy_score(all_true_labels, all_predictions), f1_score(
            all_true_labels, all_predictions
        )
    if task in ["mnli_matched", "mnli_mismatched", "mnli"]:
        return accuracy_score(all_true_labels, all_predictions), None
    raise ValueError(f"Unknown task: {task}")


def evaluate_global_model(
    global_model, dataloader, args, max_metric1, max_metric2, device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    global_model.to(device)

    global_model.eval()
    eval_loss = 0
    loss_steps = 0
    has_labels_for_metrics = True
    all_predictions = []
    all_true_labels = []

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.get("labels")
        has_invalid_labels = labels is not None and (labels < 0).any().item()
        if has_invalid_labels:
            has_labels_for_metrics = False
            model_inputs = {k: v for k, v in batch.items() if k != "labels"}
        else:
            model_inputs = batch
        with torch.no_grad():
            outputs = global_model(**model_inputs)

            if labels is not None and not has_invalid_labels and outputs.loss is not None:
                eval_loss += outputs.loss.detach().cpu().numpy()
                loss_steps += 1

            if args.task == "stsb":
                predictions = outputs.logits.squeeze().cpu().numpy()
            else:
                predictions = outputs.logits.argmax(dim=-1).cpu().numpy()
            all_predictions.extend(predictions)
            if labels is not None and not has_invalid_labels:
                all_true_labels.extend(labels.cpu().numpy())

    if loss_steps > 0:
        eval_loss /= loss_steps

    if not has_labels_for_metrics or len(all_true_labels) == 0:
        print(
            f"{args.task} - Eval Loss: {eval_loss:.4f} (no labels for metrics in this split)"
        )
        return max_metric1, max_metric2

    metric1, metric2 = calculate_metrics(all_true_labels, all_predictions, args.task)

    if metric1 > max_metric1:
        max_metric1 = metric1

    if metric2 is not None and metric2 > max_metric2:
        max_metric2 = metric2

    print(f"{args.task} - Eval Loss: {eval_loss:.4f}, Metric 1: {metric1:.4f}")
    if metric2 is not None:
        print(f"{args.task} - Metric 2: {metric2:.4f}")
    print(f"{args.task} - Max Metric 1: {max_metric1:.4f}")
    if max_metric2 is not None:
        print(f"{args.task} - Max Metric 2: {max_metric2:.4f}")

    return max_metric1, max_metric2


def get_lr_scheduler(optimizer, num_warmup_steps, num_training_steps):
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )


def train_client_e2e(model, train_dataset, val_dataset, tokenizer, args):
    num_epochs = args.local_epochs
    per_device_train_batch_size = args.batch_size
    num_training_steps = len(train_dataset) * num_epochs // per_device_train_batch_size
    num_warmup_steps = int(0.1 * num_training_steps)

    optimizer = torch.optim.AdamW(model.parameters())

    training_args = TrainingArguments(
        output_dir="./models_trained/gpt4/dump/models/gpt2-e2e-lora_gpt4",
        overwrite_output_dir=True,
        logging_dir="./models_trained/gpt4/dump/logs/gpt2-e2e-lora_gpt4",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=num_epochs,
        learning_rate=args.lr,
        weight_decay=0.01,
        label_smoothing_factor=0.1,
        run_name="fed-lora",
        logging_steps=100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        optimizers=(
            optimizer,
            get_lr_scheduler(optimizer, num_warmup_steps, num_training_steps),
        ),
    )

    trainer.train()
    return model.state_dict()


def gen_and_save(model, dataloader, tokenizer, args):
    device = args.device
    model.to(device)
    model.eval()

    all_predictions = []
    all_inputs = []
    with torch.no_grad():
        for _step, batch in enumerate(tqdm(dataloader)):
            inputs = {k: v.to(device) for k, v in batch.items()}

            generated = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=inputs["input_ids"].shape[1] + 50,
                num_return_sequences=1,
                no_repeat_ngram_size=4,
                do_sample=True,
                num_beams=10,
                penalty_alpha=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

            input_seq = tokenizer.batch_decode(
                inputs["input_ids"], skip_special_tokens=True
            )
            predictions = [
                tokenizer.decode(
                    generated[i][len(inputs["input_ids"][i]) :],
                    skip_special_tokens=True,
                )
                for i in range(generated.shape[0])
            ]

            all_inputs.extend(input_seq)
            all_predictions.extend(predictions)

    return all_predictions, all_inputs


def process_lists(input_list, second_list, third_list):
    result1 = []
    result2 = []
    result3 = []
    current_group = []
    current_item = None
    second_list_index = 0

    for item in input_list:
        if item != current_item:
            if current_group:
                result1.append(current_group)
                result2.append(current_item)
                result3.append(third_list[second_list_index - 1])
            current_item = item
            current_group = [second_list[second_list_index]]
            second_list_index += 1
        else:
            if second_list_index < len(second_list):
                current_group.append(second_list[second_list_index])
                second_list_index += 1

    if current_group:
        result1.append(current_group)

    return result1, result2, result3


def evaluate_e2e_save_text(model, test_data, tokenizer, args):
    def preprocess_function2(examples):
        inputs = examples["meaning_representation"]
        targets = examples["human_reference"]

        model_inputs = [f"{input_} ->" for input_, target in zip(inputs, targets)]

        tokenized_inputs = tokenizer(
            model_inputs,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].clone()
        tokenized_inputs["labels"][tokenized_inputs["attention_mask"] == 0] = -100

        return tokenized_inputs

    tokenized_test_dataset = test_data.map(preprocess_function2, batched=True)
    tokenized_test_dataset = tokenized_test_dataset.remove_columns(
        ["meaning_representation", "human_reference"]
    )
    tokenized_test_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )

    test_dataloader = create_dataloader(tokenized_test_dataset, args)
    all_predictions, all_inputs = gen_and_save(model, test_dataloader, tokenizer, args)
    all_references = test_data[0 : len(all_predictions)]["human_reference"]

    all_references_new, all_inputs_new, all_predictions_new = process_lists(
        all_inputs, all_references, all_predictions
    )

    path_pred = args.run_dir + "/predictions.txt"
    path_ref = args.run_dir + "/refs_exact.txt"

    if not os.path.exists(args.run_dir):
        os.makedirs(args.run_dir)

    with open(path_pred, "w") as file:
        for item in all_predictions_new:
            file.write(item.strip() + "\n")

    with open(path_ref, "w") as file:
        for str_list in all_references_new:
            for item in str_list:
                file.write(item.strip() + "\n")

            file.write("\n")
