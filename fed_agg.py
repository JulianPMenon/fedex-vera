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

def aggregate_models_vera_fedex(global_model, client_models, args):
    """
    Federated VeRA aggregation with covariance (FedEx-style).
    A and B are frozen and shared across clients.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    global_model = global_model.to(device)
    global_sd = global_model.state_dict()

    client_sds = [cm.state_dict() for cm in client_models]
    num_clients = len(client_models)

    # 1. Aggregate classifier (standard FedAvg)
    for k in global_sd:
        if "classifier" in k:
            global_sd[k] = torch.stack(
                [csd[k].float() for csd in client_sds], dim=0
            ).mean(dim=0)

    # 2. VeRA aggregation + covariance error
    for name, module in global_model.named_modules():
        if not (
            hasattr(module, "vera_A")
            and hasattr(module, "vera_B")
            and hasattr(module, "vera_lambda_b")
            and hasattr(module, "vera_lambda_d")
        ):
            continue

        A_key = name + ".vera_A.default.weight"
        B_key = name + ".vera_B.default.weight"
        lb_key = name + ".vera_lambda_b.default.weight"
        ld_key = name + ".vera_lambda_d.default.weight"

        A = global_sd[A_key]   # frozen
        B = global_sd[B_key]   # frozen

        lambda_bs = [csd[lb_key] for csd in client_sds]
        lambda_ds = [csd[ld_key] for csd in client_sds]

        # ---- averages ----
        lambda_b_avg = torch.stack(lambda_bs).mean(dim=0)
        lambda_d_avg = torch.stack(lambda_ds).mean(dim=0)

        # ---- average of full VeRA updates ----
        deltaW_avg_direct = sum(
            lambda_bs[i] @ B @ lambda_ds[i] @ A
            for i in range(num_clients)
        ) / num_clients

        # ---- VeRA update from averaged parameters ----
        deltaW_from_avg = lambda_b_avg @ B @ lambda_d_avg @ A

        # ---- covariance (FedEx-style) error ----
        covariance_error = deltaW_avg_direct - deltaW_from_avg

        # 3. Store / use the error
        # (A) Log it
        if hasattr(args, "log_covariance") and args.log_covariance:
            err_norm = torch.norm(covariance_error, p="fro").item()
            print(f"[VeRA-FedEx] {name} covariance error: {err_norm:.6f}")

        # (B) Optional FedEx-style correction
        if hasattr(args, "fedex") and args.fedex:
            # project error back into base weight space
            base_key = name + ".base_layer.weight"
            if base_key in global_sd:
                global_sd[base_key] += args.fedex_lr * covariance_error

        # 4. Update global lambdas (canonical VeRA-FL)
        global_sd[lb_key] = lambda_b_avg
        global_sd[ld_key] = lambda_d_avg

        # push back to clients
        for cm, csd in zip(client_models, client_sds):
            csd[lb_key] = lambda_b_avg
            csd[ld_key] = lambda_d_avg
            cm.load_state_dict(csd, strict=False)

    global_model.load_state_dict(global_sd, strict=False)
    return global_model
