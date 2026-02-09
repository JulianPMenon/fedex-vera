import torch
from torch.utils.data import DataLoader
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from peft import get_peft_model, LoraConfig, TaskType, VeraConfig
from data_utils import *
from models import *
import argparse
import warnings
import os
from datetime import datetime
import numpy as np
import wandb
from train_eval import *
from fed_agg import *
import json
from utils import *

parser = argparse.ArgumentParser(description="Federated Learning with LoRA")

parser.add_argument(
    "--task", type=str, default="cola", help="GLUE task to fine-tune on"
)
parser.add_argument("--model", type=str, default="roberta-base", help="Model name")
parser.add_argument("--r", type=int, default=4, help="Rank for LoRA/VeRA (replaces lora_r)")
parser.add_argument("--lora_alpha", type=int, default=8, help="LoRA/VeRA alpha value")
parser.add_argument(
    "--lora_dropout", type=float, default=0.1, help="LoRA dropout value"
)
parser.add_argument('--vera', action='store_true', help='Use VeRA adaptation')
parser.add_argument('--d_initial', type=float, default=0.1, help='Initial value for d in VeRA')
parser.add_argument('--vera_dropout', type=float, default=0.0, help='VeRA dropout value')
parser.add_argument('--projection_prng_key', type=int, default=0, help='Projection PRNG key for VeRA')
parser.add_argument('--save_projection', type=bool, default=True, help='Save projection in VeRA')
parser.add_argument('--fan_in_fan_out', type=bool, default=False, help='Fan-in fan-out for VeRA')
parser.add_argument('--bias', type=str, default='none', help='Bias type for VeRA')
parser.add_argument('--modules_to_save', type=str, default=None, help='Modules to save for VeRA')
parser.add_argument('--init_weights', type=bool, default=True, help='Init weights for VeRA')
parser.add_argument('--layers_to_transform', type=str, default=None, help='Layers to transform for VeRA')
parser.add_argument('--layers_pattern', type=str, default=None, help='Layers pattern for VeRA')
parser.add_argument("--rslora", action="store_true", help="Use RSLoRA")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
parser.add_argument(
    "--agg_type", type=str, default="ours", help="Type of aggregation"
)
parser.add_argument("--num_clients", type=int, default=3, help="Number of clients")
parser.add_argument("--rounds", type=int, default=50, help="Number of rounds")
parser.add_argument(
    "--local_epochs", type=int, default=3, help="Number of local epochs"
)
parser.add_argument("--warmup_ratio", type=float, default=0.06, help="Warmup ratio")
parser.add_argument(
    "--max_seq_length", type=int, default=512, help="Maximum sequence length"
)
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--seed", type=int, default=42, help="Random seed")

args = parser.parse_args()

#wandb.init(project="project_name", config=args)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


def federated_learning(task):

    train_data, val_data, test_data = load_and_preprocess_data(task)

    num_labels = len(set(train_data["labels"]))

    if args.task == "stsb":
        num_labels = 1

    client_dataloaders = create_client_dataloaders(train_data, args)
    val_dataloader = create_dataloader(val_data, args)

    max_metric_1 = 0
    max_metric_2 = 0

    if args.agg_type == "ffa":
        global_model = create_peft_FFA_model(num_labels, args)
    else:
        global_model = create_peft_model(num_labels, args)

    if args.vera and args.agg_type == "ours_vera":
        vera_keys = [k for k in global_model.state_dict().keys() if "vera_lambda" in k]
        print("VeRA lambda keys sample:", vera_keys[:4])
        print("VeRA lambda key count:", len(vera_keys))

    client_models = []

    for i in range(args.num_clients):
        if args.agg_type == "ffa":
            client_model = create_peft_FFA_model(num_labels, args)
        else:
            client_model = create_peft_model(num_labels, args)
        client_models.append(client_model)

    for round in range(args.rounds):
        print(f"Round {round + 1}/{args.rounds}")

        client_model_state_dicts = []
        for i in range(args.num_clients):
            client_model = client_models[i]
            client_model.load_state_dict(global_model.state_dict())
            client_model_state_dict = train_client(
                client_model, client_dataloaders[i], args
            )
            client_model_state_dicts.append(client_model_state_dict)

        if args.agg_type == "normal":
            global_model = aggregate_models_normal(global_model, client_models)
        elif args.agg_type == "ours":
            global_model = aggregate_models_ours(global_model, client_models, args)
        elif args.agg_type == "ours_vera":
            global_model = aggregate_models_ours_vera_fedex(global_model, client_models, args)
        elif args.agg_type == "ffa":
            global_model = aggregate_models_ffa(global_model, client_models)

        max_metric_1, max_metric_2 = evaluate_global_model(
            global_model, val_dataloader, args, max_metric_1, max_metric_2
        )


# Main execution
if __name__ == "__main__":
    task = args.task
    model = federated_learning(task)
