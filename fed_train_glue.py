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
import torch.multiprocessing as mp
import traceback
import tempfile
import shutil


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
parser.add_argument("--sorf_seed", type=int, default=42, help="Seed for SORF matrix D1/D2/D3")
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
parser.add_argument(
    "--parallel_clients",
    action="store_true",
    help="Train clients in parallel with multiprocessing",
)
parser.add_argument(
    "--server_gpu",
    type=int,
    default=0,
    help="GPU id for server aggregation/evaluation",
)
parser.add_argument(
    "--client_gpus",
    type=str,
    default=None,
    help="Comma-separated GPU ids for client training",
)

args = parser.parse_args()

#wandb.init(project="project_name", config=args)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


def _parse_client_gpus(args):
    if args.client_gpus:
        return [int(x.strip()) for x in args.client_gpus.split(",") if x.strip()]
    return []


def _is_base_weight_key(key):
    return key.endswith("base_layer.weight") or key.endswith("base_layer.bias")


def _is_classifier_key(key):
    return "classifier" in key


def _is_vera_lambda_key(key):
    return "vera_lambda_b" in key or "vera_lambda_d" in key


def _is_sorf_param_key(key):
    return any(x in key for x in ("sorf_b_real", "sorf_b_imag", "sorf_d_scale", "sorf_d_angles"))


def _is_sorf_d_key(key):
    return "sorf_d_scale" in key or "sorf_d_angles" in key


def _get_client_broadcast_state(model, args):
    state = model.state_dict()
    if args.agg_type == "sorf_vera":
        return {
            k: v
            for k, v in state.items()
            if _is_sorf_param_key(k) or _is_base_weight_key(k) or _is_classifier_key(k)
        }
    if getattr(args, "vera", False):
        return {
            k: v
            for k, v in state.items()
            if _is_vera_lambda_key(k) or _is_base_weight_key(k) or _is_classifier_key(k)
        }
    return state


def _get_client_upload_state(state_dict, args):
    if args.agg_type == "sorf_vera":
        return {
            k: v
            for k, v in state_dict.items()
            if _is_sorf_d_key(k) or _is_classifier_key(k)
        }
    if getattr(args, "vera", False):
        return {
            k: v
            for k, v in state_dict.items()
            if _is_vera_lambda_key(k) or _is_classifier_key(k)
        }
    return state_dict


def _get_tmp_dir():
    return (
        os.environ.get("SLURM_TMPDIR")
        or os.environ.get("TMPDIR")
        or "/tmp"
    )


def _train_client_worker(
    client_id,
    gpu_id,
    model_state_dict,
    client_data,
    args,
    num_labels,
    return_queue,
    save_path,
):
    try:
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
            device = torch.device(f"cuda:{gpu_id}")
        else:
            device = torch.device("cpu")

        if args.agg_type == "ffa":
            client_model = create_peft_FFA_model(num_labels, args)
        elif args.agg_type == "sorf_vera":
            client_model = create_peft_sorf_vera_model(num_labels, args)
        else:
            client_model = create_peft_model(num_labels, args)

        client_model.load_state_dict(model_state_dict, strict=False)
        client_loader = DataLoader(
            client_data, batch_size=args.batch_size, shuffle=True
        )
        client_state = train_client(client_model, client_loader, args, device=device)
        upload_state = _get_client_upload_state(client_state, args)
        cpu_state = {k: v.detach().cpu() for k, v in upload_state.items()}
        tmp_path = f"{save_path}.tmp"
        torch.save(cpu_state, tmp_path, _use_new_zipfile_serialization=False)
        os.replace(tmp_path, save_path)
        return_queue.put({"ok": True, "client_id": client_id, "path": save_path})
    except Exception as exc:
        if "tmp_path" in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)
        return_queue.put(
            {
                "ok": False,
                "client_id": client_id,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )

def federated_learning(task):

    train_data, val_data, test_data = load_and_preprocess_data(task)

    num_labels = len(set(train_data["labels"]))

    if args.task == "stsb":
        num_labels = 1

    client_gpu_ids = []
    use_parallel = args.parallel_clients
    if use_parallel and not torch.cuda.is_available():
        print("CUDA not available; falling back to sequential client training.")
        use_parallel = False

    if use_parallel:
        client_gpu_ids = _parse_client_gpus(args)
        if not client_gpu_ids:
            total_gpus = torch.cuda.device_count()
            candidate = [i for i in range(total_gpus) if i != args.server_gpu]
            client_gpu_ids = candidate[: args.num_clients]
        if len(client_gpu_ids) < args.num_clients:
            raise ValueError(
                "Not enough client GPUs for parallel training. "
                f"Needed {args.num_clients}, got {len(client_gpu_ids)}."
            )
        if args.server_gpu in client_gpu_ids:
            print("Warning: server GPU also used for client training.")
        client_datasets = create_client_datasets(train_data, args)
    else:
        client_dataloaders = create_client_dataloaders(train_data, args)
    val_dataloader = create_dataloader(val_data, args)
    test_dataloader = create_dataloader(test_data, args)

    max_metric_1 = 0
    max_metric_2 = 0

    if torch.cuda.is_available():
        torch.cuda.set_device(args.server_gpu)
    server_device = (
        torch.device(f"cuda:{args.server_gpu}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    if args.agg_type == "ffa":
        global_model = create_peft_FFA_model(num_labels, args)
    elif args.agg_type == "sorf_vera":
        global_model = create_peft_sorf_vera_model(num_labels, args)
    else:
        global_model = create_peft_model(num_labels, args)
    global_model.to(server_device)

    if args.vera and args.agg_type == "ours_vera":
        vera_keys = [k for k in global_model.state_dict().keys() if "vera_lambda" in k]
        print("VeRA lambda keys sample:", vera_keys[:4])
        print("VeRA lambda key count:", len(vera_keys))

    client_models = []

    for i in range(args.num_clients):
        if args.agg_type == "ffa":
            client_model = create_peft_FFA_model(num_labels, args)
        elif args.agg_type == "sorf_vera":
            client_model = create_peft_sorf_vera_model(num_labels, args)
        else:
            client_model = create_peft_model(num_labels, args)
        client_models.append(client_model)

    for round in range(args.rounds):
        print(f"Round {round + 1}/{args.rounds}")

        client_model_state_dicts = []
        if use_parallel:
            broadcast_state = _get_client_broadcast_state(global_model, args)
            global_state_cpu = {k: v.detach().cpu() for k, v in broadcast_state.items()}
            return_queue = mp.Queue()
            processes = []
            round_dir = tempfile.mkdtemp(
                prefix=f"fed_clients_round_{round + 1}_",
                dir=_get_tmp_dir(),
            )
            for i in range(args.num_clients):
                save_path = os.path.join(round_dir, f"client_{i}.pt")
                p = mp.Process(
                    target=_train_client_worker,
                    args=(
                        i,
                        client_gpu_ids[i],
                        global_state_cpu,
                        client_datasets[i],
                        args,
                        num_labels,
                        return_queue,
                        save_path,
                    ),
                )
                p.start()
                processes.append(p)

            client_state_map = {}
            for _ in range(args.num_clients):
                msg = return_queue.get()
                if not msg.get("ok", False):
                    for p in processes:
                        if p.is_alive():
                            p.terminate()
                    error_text = msg.get("error", "unknown error")
                    error_tb = msg.get("traceback", "")
                    raise RuntimeError(
                        f"Client {msg.get('client_id')} failed: {error_text}\n{error_tb}"
                    )
                client_state_map[msg["client_id"]] = msg["path"]

            for p in processes:
                p.join()

            client_models = []
            for i in range(args.num_clients):
                if args.agg_type == "ffa":
                    client_model = create_peft_FFA_model(num_labels, args)
                elif args.agg_type == "sorf_vera":
                    client_model = create_peft_sorf_vera_model(num_labels, args)
                else:
                    client_model = create_peft_model(num_labels, args)
                state_path = client_state_map[i]
                client_state = torch.load(state_path, map_location="cpu", weights_only=True)
                client_model.load_state_dict(client_state, strict=False)
                client_model.to(server_device)
                client_models.append(client_model)
            shutil.rmtree(round_dir, ignore_errors=True)
        else:
            for i in range(args.num_clients):
                client_model = client_models[i]
                broadcast_state = _get_client_broadcast_state(global_model, args)
                client_model.load_state_dict(broadcast_state, strict=False)
                client_model_state_dict = train_client(
                    client_model, client_dataloaders[i], args, device=server_device
                )
                client_model_state_dicts.append(client_model_state_dict)
            for client_model in client_models:
                client_model.to(server_device)

        if args.agg_type == "normal":
            global_model = aggregate_models_normal(global_model, client_models)
        elif args.agg_type == "ours":
            global_model = aggregate_models_ours(global_model, client_models, args)
        elif args.agg_type == "ours_vera":
            global_model = aggregate_models_ours_vera_fedex(global_model, client_models, args)
        elif args.agg_type == "sorf_vera":
            global_model = aggregate_models_sorf_vera(global_model, client_models, args)
        elif args.agg_type == "ffa":
            global_model = aggregate_models_ffa(global_model, client_models)

        
        if round == args.rounds - 1:
            print("Testing final global model")
            max_metric_1, max_metric_2 = evaluate_global_model(
                global_model,
                test_dataloader,
                args,
                max_metric_1,
                max_metric_2,
                device=server_device,
            )
        else:
            max_metric_1, max_metric_2 = evaluate_global_model(
                global_model,
                val_dataloader,
                args,
                max_metric_1,
                max_metric_2,
                device=server_device,
            )



# Main execution
if __name__ == "__main__":
    if args.parallel_clients:
        try:
            mp.set_start_method("spawn")
        except RuntimeError:
            pass
    task = args.task
    model = federated_learning(task)
