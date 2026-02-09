import torch
import numpy as np


# --------------------------------------------------
# Helper
# --------------------------------------------------

def _get_state_dicts(client_models):
    return [m.state_dict() for m in client_models]


# --------------------------------------------------
# 1. Normal aggregation (LoRA + VeRA + classifier)
# --------------------------------------------------

def aggregate_models_normal(global_model, client_models):

    global_state = global_model.state_dict()
    client_states = _get_state_dicts(client_models)
    num_clients = len(client_models)

    for k in global_state.keys():
        if (
            "lora" in k
            or "vera_lambda" in k
            or "classifier" in k
        ):
            if all(k in cs for cs in client_states):
                global_state[k] = torch.stack(
                    [cs[k].float() for cs in client_states],
                    dim=0
                ).mean(0)

    global_model.load_state_dict(global_state, strict=False)
    return global_model


# --------------------------------------------------
# 2. FFA aggregation (LoRA-B / VeRA-b only)
# --------------------------------------------------

def aggregate_models_ffa(global_model, client_models):

    global_state = global_model.state_dict()
    client_states = _get_state_dicts(client_models)
    num_clients = len(client_models)

    for k in global_state.keys():
        if (
            "lora_B" in k
            or "vera_lambda_b" in k
            or "classifier" in k
        ):
            if all(k in cs for cs in client_states):
                global_state[k] = torch.stack(
                    [cs[k].float() for cs in client_states],
                    dim=0
                ).mean(0)

    global_model.load_state_dict(global_state, strict=False)
    return global_model


# --------------------------------------------------
# 3. Our LoRA aggregation (FedAvg + residue)
# --------------------------------------------------

def aggregate_models_ours(global_model, client_models, args):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    global_model = global_model.to(device)

    global_state = global_model.state_dict()
    client_states = _get_state_dicts(client_models)
    num_clients = len(client_models)

    # ---- classifier ----
    for k in global_state.keys():
        if "classifier" in k and all(k in cs for cs in client_states):
            global_state[k] = torch.stack(
                [cs[k].float() for cs in client_states],
                dim=0
            ).mean(0)

    # ---- LoRA FedEx-style ----
    for name, module in global_model.named_modules():

        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):

            A_key = f"{name}.lora_A.default.weight"
            B_key = f"{name}.lora_B.default.weight"
            base_key = f"{name}.base_layer.weight"

            if not all(
                A_key in cs and B_key in cs for cs in client_states
            ):
                continue

            A_ws = torch.stack([cs[A_key] for cs in client_states]).to(device)
            B_ws = torch.stack([cs[B_key] for cs in client_states]).to(device)

            M = sum(B_ws[i] @ A_ws[i] for i in range(num_clients)) / num_clients

            A_avg = A_ws.mean(0)
            B_avg = B_ws.mean(0)

            scaling = (
                args.lora_alpha / np.sqrt(args.r)
                if getattr(args, "rslora", False)
                else args.lora_alpha / args.r
            )

            residue = M - (B_avg @ A_avg)

            global_state[A_key] = A_avg
            global_state[B_key] = B_avg
            global_state[base_key] += residue.T * scaling

    global_model.load_state_dict(global_state, strict=False)
    return global_model


# --------------------------------------------------
# 4. VeRA + FedEx (FULLY FIXED)
# --------------------------------------------------

def aggregate_models_ours_vera_fedex(global_model, client_models, args):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    global_model = global_model.to(device)

    global_state = global_model.state_dict()
    client_states = _get_state_dicts(client_models)
    num_clients = len(client_models)

    # ---- classifier ----
    for k in global_state.keys():
        if "classifier" in k and all(k in cs for cs in client_states):
            global_state[k] = torch.stack(
                [cs[k].float() for cs in client_states],
                dim=0
            ).mean(0)

    # ---- VeRA FedEx ----
    for name, module in global_model.named_modules():

        if (
            hasattr(module, "vera_A")
            and hasattr(module, "vera_B")
            and hasattr(module, "vera_lambda_b")
            and hasattr(module, "vera_lambda_d")
        ):

            # PEFT may store VeRA lambdas with or without `.default` and/or `.weight`.
            lb_candidates = (
                f"{name}.vera_lambda_b.weight",
                f"{name}.vera_lambda_b.default.weight",
                f"{name}.vera_lambda_b.default",
                f"{name}.vera_lambda_b",
            )
            ld_candidates = (
                f"{name}.vera_lambda_d.weight",
                f"{name}.vera_lambda_d.default.weight",
                f"{name}.vera_lambda_d.default",
                f"{name}.vera_lambda_d",
            )

            lb_key = next((k for k in lb_candidates if k in global_state), None)
            ld_key = next((k for k in ld_candidates if k in global_state), None)

            if lb_key is None or ld_key is None:
                continue
            base_key = f"{name}.base_layer.weight"

            # critical: all clients must have these
            if not all(
                lb_key in cs and ld_key in cs for cs in client_states
            ):
                continue

            # frozen shared matrices
            A = module.vera_A.default
            B = module.vera_B.default

            def _as_diag_matrix(tensor, size):
                if tensor.dim() == 0:
                    return torch.eye(size, device=device, dtype=tensor.dtype) * tensor
                if tensor.dim() == 1:
                    return torch.diag(tensor)
                return tensor

            lambda_bs_raw = [cs[lb_key].detach() for cs in client_states]
            lambda_ds_raw = [cs[ld_key].detach() for cs in client_states]

            b_size = B.shape[0]
            d_size = B.shape[1]

            lambda_bs = torch.stack(
                [_as_diag_matrix(lb, b_size) for lb in lambda_bs_raw]
            ).to(device)

            lambda_ds = torch.stack(
                [_as_diag_matrix(ld, d_size) for ld in lambda_ds_raw]
            ).to(device)

            # FedEx core
            M = sum(
                lambda_bs[i] @ B @ lambda_ds[i] @ A
                for i in range(num_clients)
            ) / num_clients

            lambda_b_avg = torch.stack(lambda_bs_raw).mean(0)
            lambda_d_avg = torch.stack(lambda_ds_raw).mean(0)

            residue = M - (lambda_b_avg @ B @ lambda_d_avg @ A)

            global_state[lb_key] = lambda_b_avg
            global_state[ld_key] = lambda_d_avg

            if getattr(args, "fedex", False):
                global_state[base_key] += args.fedex_lr * residue

    global_model.load_state_dict(global_state, strict=False)
    return global_model
