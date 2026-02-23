import torch
import numpy as np
import time
from functools import wraps
from sorf_vera import fit_b, fit_d, build_lambda_d_rot, build_gamma

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} executed in {end - start:.6f} seconds")
        return result
    return wrapper


# --------------------------------------------------
# Helper
# --------------------------------------------------

def _get_state_dicts(client_models):
    return [m.state_dict() for m in client_models]


# --------------------------------------------------
# 1. Normal aggregation (LoRA + VeRA + classifier)
# --------------------------------------------------

@timer
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

@timer
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

@timer
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

@timer
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

            lambda_b_avg_mat = _as_diag_matrix(lambda_b_avg, b_size)
            lambda_d_avg_mat = _as_diag_matrix(lambda_d_avg, d_size)

            residue = M - (lambda_b_avg_mat @ B @ lambda_d_avg_mat @ A)

            global_state[lb_key] = lambda_b_avg
            global_state[ld_key] = lambda_d_avg

            if getattr(args, "fedex", False):
                global_state[base_key] += args.fedex_lr * residue

    global_model.load_state_dict(global_state, strict=False)
    return global_model


# --------------------------------------------------
# 5. SORF-VeRA aggregation (exact refit)
# --------------------------------------------------

def aggregate_models_sorf_vera(global_model, client_models, args):

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

    # ---- SORF-VeRA exact refit ----
    for name, module in global_model.named_modules():

        if not hasattr(module, 'sorf_d_scale'):
            continue

        d_out = module.out_features
        d_in = module.in_features
        r = module.r

        # State dict keys for SORF params
        scale_key = f"{name}.sorf_d_scale"
        angles_key = f"{name}.sorf_d_angles"
        b_real_key = f"{name}.sorf_b_real"
        b_imag_key = f"{name}.sorf_b_imag"

        if not all(
            scale_key in cs and angles_key in cs
            for cs in client_states
        ):
            continue

        # Get current b from global model (same across clients, frozen)
        b_real = global_state[b_real_key].to(device)
        b_imag = global_state[b_imag_key].to(device)

        # Shared A, B matrices — slice to this layer's dims
        A = module.vera_A['default'][:r, :d_in].to(device).float()
        B = module.vera_B['default'][:d_out, :r].to(device).float()

        # SORF S matrix
        S = module.sorf_S.to(device).float()

        # Average Lambda_d matrices across clients (not the raw params,
        # because Givens rotations are nonlinear in angles)
        avg_Ld = sum(
            build_lambda_d_rot(
                cs[scale_key].to(device).float(),
                cs[angles_key].to(device).float(),
                r, device, torch.float32
            )
            for cs in client_states
        ) / num_clients

        # Compute full DeltaW from averaged Lambda_d
        Gamma_b = build_gamma(b_real.float(), b_imag.float())
        DeltaW = S @ Gamma_b @ B @ avg_Ld @ A

        # Rank-r truncated SVD
        U, S_vals, Vh = torch.linalg.svd(DeltaW, full_matrices=False)
        U_r = U[:, :r]          # (d_out, r)
        Sigma_r = S_vals[:r]    # (r,)
        V_r = Vh[:r, :].T      # (d_in, r)

        # Fit new b (closed-form per-block lstsq)
        b_real_new, b_imag_new = fit_b(S, B, U_r, Sigma_r, d_out, r)

        # Fit new d (pseudoinverse + Givens decomposition)
        scale_new, angles_new = fit_d(A, V_r, r)

        # Update global state
        global_state[b_real_key] = b_real_new
        global_state[b_imag_key] = b_imag_new
        global_state[scale_key] = scale_new
        global_state[angles_key] = angles_new

    global_model.load_state_dict(global_state, strict=False)
    return global_model
