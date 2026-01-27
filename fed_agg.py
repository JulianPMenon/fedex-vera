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

def aggregate_models_ours_vera(global_model, client_models, args):

    global_model = (
        global_model.to("cuda") if torch.cuda.is_available() else global_model
    )
    global_dict = global_model.state_dict()

    # Extract state dicts from client models
    client_state_dicts = [cm.state_dict() for cm in client_models]

    # Aggregate classifier weights as before
    for k in global_dict.keys():
        if "classifier" in k:
            global_dict[k] = torch.stack(
                [client_sd.get(k, global_dict[k]).float() for client_sd in client_state_dicts], 0
            ).mean(0)

    # Update each client's classifier weights to the aggregated value
    for cm, client_sd in zip(client_models, client_state_dicts):
        for k in global_dict.keys():
            if "classifier" in k:
                client_sd[k] = global_dict[k]
        cm.load_state_dict(client_sd)

    # VeRA aggregation using vera_lambda_b and vera_lambda_d (diagonal matrices)
    for name, module in global_model.named_modules():
        if hasattr(module, "vera_A") and hasattr(module, "vera_B"):
            vera_A_key = name + ".vera_A.default.weight"
            vera_B_key = name + ".vera_B.default.weight"
            vera_lambda_b_key = name + ".vera_lambda_b.default.weight"
            vera_lambda_d_key = name + ".vera_lambda_d.default.weight"
            base_layer_key = name + ".base_layer.weight"

            vera_A = global_dict.get(vera_A_key, None)
            vera_B = global_dict.get(vera_B_key, None)
            if vera_A is None or vera_B is None:
                continue

            # Collect lambda_b and lambda_d diagonal matrices from clients
            lambda_b_matrices = []
            lambda_d_matrices = []
            for client_sd in client_state_dicts:
                if vera_lambda_b_key in client_sd and vera_lambda_d_key in client_sd:
                    lambda_b_matrices.append(client_sd[vera_lambda_b_key].detach())
                    lambda_d_matrices.append(client_sd[vera_lambda_d_key].detach())

            if len(lambda_b_matrices) == 0 or len(lambda_d_matrices) == 0:
                raise ValueError(f"No client models have {vera_lambda_b_key} or {vera_lambda_d_key} for aggregation.")

            lambda_b_avg = torch.stack(lambda_b_matrices).mean(0)
            lambda_d_avg = torch.stack(lambda_d_matrices).mean(0)

            # Compute VeRA update: M = vera_B @ lambda_b_avg + vera_A @ lambda_d_avg
            M = vera_B @ lambda_b_avg + vera_A @ lambda_d_avg

            scaling_factor = (
                args.lora_alpha / np.sqrt(args.r)
                if hasattr(args, "rslora") and args.rslora
                else args.lora_alpha / args.r
            )

            if base_layer_key in global_dict:
                global_dict[base_layer_key] += torch.transpose(M * scaling_factor, 1, 0)

            # Optionally, update global lambda_b and lambda_d to aggregated values
            global_dict[vera_lambda_b_key] = lambda_b_avg
            global_dict[vera_lambda_d_key] = lambda_d_avg

            # Optionally, update each client's lambda_b and lambda_d to aggregated values
            for cm, client_sd in zip(client_models, client_state_dicts):
                if vera_lambda_b_key in client_sd:
                    client_sd[vera_lambda_b_key] = lambda_b_avg
                if vera_lambda_d_key in client_sd:
                    client_sd[vera_lambda_d_key] = lambda_d_avg
                cm.load_state_dict(client_sd)

    global_model.load_state_dict(global_dict)
    return global_model