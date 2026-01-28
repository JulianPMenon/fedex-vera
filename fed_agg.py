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
from sklearn.metrics import matthews_corrcoef
import numpy as np
import torch.nn as nn


def aggregate_models_normal(global_model, client_models):

    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        if "lora" in k:  # Only aggregate LoRA parameters
            global_dict[k] = torch.stack(
                [client_models[i][k].float() for i in range(len(client_models))], 0
            ).mean(0)

        if "vera" in k:  # Only aggregate Vera parameters
            global_dict[k] = torch.stack(
                [client_models[i][k].float() for i in range(len(client_models))], 0
            ).mean(0)

        if "classifier" in k:
            global_dict[k] = torch.stack(
                [client_models[i][k].float() for i in range(len(client_models))], 0
            ).mean(0)

    global_model.load_state_dict(global_dict)

    return global_model

    


def aggregate_models_ffa(global_model, client_models):

    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        if "lora_B" in k:  # Only aggregate LoRA B parameters
            global_dict[k] = torch.stack(
                [client_models[i][k].float() for i in range(len(client_models))], 0
            ).mean(0)

        if "vera_b" in k: # Only aggregate Vera b parameters
            global_dict[k] = torch.stack(
                [client_models[i][k].float() for i in range(len(client_models))], 0
            ).mean(0)

        if "classifier" in k:
            global_dict[k] = torch.stack(
                [client_models[i][k].float() for i in range(len(client_models))], 0
            ).mean(0)

    global_model.load_state_dict(global_dict)

    return global_model


def aggregate_models_ours(global_model, client_models, args):

    global_model = (
        global_model.to("cuda") if torch.cuda.is_available() else global_model
    )
    global_dict = global_model.state_dict()
    for k in global_dict.keys():

        if "classifier" in k:
            global_dict[k] = torch.stack(
                [client_models[i][k].float() for i in range(len(client_models))], 0
            ).mean(0)

    for client_model in client_models:

        for k in global_dict.keys():

            if "classifier" in k:
                client_model[k] = global_dict[k]

    for name, module in global_model.named_modules():

        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):

            lora_A_keys = name + ".lora_A.default.weight"
            lora_B_keys = name + ".lora_B.default.weight"
            base_layer_keys = name + ".base_layer.weight"

            lora_A_weights = torch.stack(
                [client_model[lora_A_keys].detach() for client_model in client_models]
            )
            lora_B_weights = torch.stack(
                [client_model[lora_B_keys].detach() for client_model in client_models]
            )

            # M shape: (d, k)
            M = sum(
                lora_B_weights[i] @ lora_A_weights[i] for i in range(len(client_models))
            ) / len(client_models)

            lora_A_avg = lora_A_weights.mean(0)
            lora_B_avg = lora_B_weights.mean(0)

            scaling_factor = (
                args.lora_alpha / np.sqrt(args.r)
                if args.rslora
                else args.lora_alpha / args.r
            )

            residue = M - lora_B_avg @ lora_A_avg

            global_dict[name + ".lora_A.default.weight"] = lora_A_avg
            global_dict[name + ".lora_B.default.weight"] = lora_B_avg
            global_dict[name + ".base_layer.weight"] += torch.transpose(
                residue * scaling_factor, 1, 0
            )
            
    global_model.load_state_dict(global_dict)

    return global_model

def aggregate_models_ours_vera_fedex(global_model, client_models, args):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    global_model = global_model.to(device)

    num_clients = len(client_models)

    # Collect state_dicts once (IMPORTANT)
    client_states = [m.state_dict() for m in client_models]
    global_state = global_model.state_dict()

    # --------------------------------------------------
    # 1. Aggregate classifier
    # --------------------------------------------------

    for k in global_state.keys():
        if "classifier" in k:
            global_state[k] = torch.stack(
                [client_states[i][k].float() for i in range(num_clients)],
                dim=0
            ).mean(0)

    # Load into global model
    global_model.load_state_dict(global_state, strict=False)

    # Sync classifier back to clients
    for client_model in client_models:
        client_sd = client_model.state_dict()
        for k in global_state.keys():
            if "classifier" in k:
                client_sd[k] = global_state[k].clone()
        client_model.load_state_dict(client_sd, strict=False)

    # --------------------------------------------------
    # 2. VeRA FedEx-style aggregation
    # --------------------------------------------------

    for name, module in global_model.named_modules():

        if (
            hasattr(module, "vera_A")
            and hasattr(module, "vera_B")
            and hasattr(module, "vera_lambda_b")
            and hasattr(module, "vera_lambda_d")
        ):

            lb_key = f"{name}.vera_lambda_b.default.weight"
            ld_key = f"{name}.vera_lambda_d.default.weight"
            base_key = f"{name}.base_layer.weight"

            # ---- FIXED GUARD (this is the missing fix) ----
            if not all(lb_key in cs and ld_key in cs for cs in client_states):
                continue

            A = module.vera_A.default
            B = module.vera_B.default

            lambda_bs = torch.stack(
                [cs[lb_key].detach() for cs in client_states]
            ).to(A.device)

            lambda_ds = torch.stack(
                [cs[ld_key].detach() for cs in client_states]
            ).to(A.device)

            M = sum(
                lambda_bs[i] @ B @ lambda_ds[i] @ A
                for i in range(num_clients)
            ) / num_clients

            lambda_b_avg = lambda_bs.mean(0)
            lambda_d_avg = lambda_ds.mean(0)

            residue = M - (lambda_b_avg @ B @ lambda_d_avg @ A)

            global_state[lb_key] = lambda_b_avg
            global_state[ld_key] = lambda_d_avg

            if getattr(args, "fedex", False):
                global_state[base_key] += args.fedex_lr * residue



    # --------------------------------------------------
    # 3. Load final global model
    # --------------------------------------------------

    global_model.load_state_dict(global_state, strict=False)

    return global_model

