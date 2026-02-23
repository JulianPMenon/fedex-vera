import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
)
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from peft import (
    get_peft_model,
    AdaLoraModel,
    AdaLoraConfig,
    TaskType,
    LoraConfig,
    prepare_model_for_kbit_training,
    VeraConfig,
)
from data_utils import *
import argparse
import types
from copy import deepcopy
import torch.nn as nn
import torch.nn.functional as F

from sorf_vera import (
    build_sorf_matrix,
    build_lambda_d_rot,
    apply_gamma_transpose_batched,
    init_sorf_vera_params,
)


def create_peft_model(num_labels, args):

    model = RobertaForSequenceClassification.from_pretrained(
        args.model, num_labels=num_labels
    )

    vera = getattr(args, 'vera', False)
    if vera:
        peft_config = VeraConfig(
            task_type=TaskType.SEQ_CLS,
            r=getattr(args, 'r', 256),
            target_modules=["query", "value"],
            projection_prng_key=getattr(args, 'projection_prng_key', 0),
            save_projection=getattr(args, 'save_projection', True),
            vera_dropout=getattr(args, 'vera_dropout', 0.0),
            d_initial=getattr(args, 'd_initial', 0.1),
            fan_in_fan_out=getattr(args, 'fan_in_fan_out', False),
            bias=getattr(args, 'bias', 'none'),
            modules_to_save=getattr(args, 'modules_to_save', None),
            init_weights=getattr(args, 'init_weights', True),
            layers_to_transform=getattr(args, 'layers_to_transform', None),
            layers_pattern=getattr(args, 'layers_pattern', None),
            inference_mode=False,
        )
    else:
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=args.r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            use_rslora=args.rslora,
            target_modules=["query", "value"],
        )

    model = get_peft_model(model, peft_config)
    return model


def create_peft_FFA_model(num_labels, args):

    model = RobertaForSequenceClassification.from_pretrained(
        args.model, num_labels=num_labels
    )

    vera = getattr(args, 'vera', False)
    if vera:
        peft_config = VeraConfig(
            task_type=TaskType.SEQ_CLS,
            r=getattr(args, 'r', 256),
            target_modules=["query", "value"],
            projection_prng_key=getattr(args, 'projection_prng_key', 0),
            save_projection=getattr(args, 'save_projection', True),
            vera_dropout=getattr(args, 'vera_dropout', 0.0),
            d_initial=getattr(args, 'd_initial', 0.1),
            fan_in_fan_out=getattr(args, 'fan_in_fan_out', False),
            bias=getattr(args, 'bias', 'none'),
            modules_to_save=getattr(args, 'modules_to_save', None),
            init_weights=getattr(args, 'init_weights', True),
            layers_to_transform=getattr(args, 'layers_to_transform', None),
            layers_pattern=getattr(args, 'layers_pattern', None),
            inference_mode=False,
        )
    else:
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=args.r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            use_rslora=args.rslora,
            target_modules=["query", "value"],
        )
    model = get_peft_model(model, peft_config)

    # Make LoRA A matrices non-trainable
    for name, param in model.named_parameters():
        if "lora_A" in name:
            param.requires_grad = False

    return model


def _sorf_vera_forward(self, x, *args, **kwargs):
    """Monkey-patched forward for SORF-VeRA modules.

    Replaces VeRA's forward with:
        h = x @ A^T -> h = h @ Ld_rot^T -> h = h @ B^T
        -> apply Gamma^T (vectorized) -> h = h @ S^T -> result += h
    """
    # Original linear (base layer)
    result = self.base_layer(x, *args, **kwargs)

    # Get shared VeRA matrices, sliced to this layer's dimensions
    A = self.vera_A['default']   # (r_full, d_in_max)
    B = self.vera_B['default']   # (d_out_max, r_full)

    d_out = self.out_features
    d_in = self.in_features
    r = self.r

    # Slice to this layer's actual dims
    sliced_A = A[:r, :d_in]   # (r, d_in)
    sliced_B = B[:d_out, :r]  # (d_out, r)

     # SORF-VeRA forward path
    # Step 1: h = x @ sliced_A^T  ->  (batch, seq, r)
    h = F.linear(x, sliced_A)

    # Step 2: h = h @ Ld_rot^T    ->  (batch, seq, r)
    Ld_rot = build_lambda_d_rot(
        self.sorf_d_scale, self.sorf_d_angles, r,
        device=x.device, dtype=x.dtype
    )
    h = h @ Ld_rot.T

    # Step 3: h = h @ sliced_B^T  ->  (batch, seq, d_out)
    h = F.linear(h, sliced_B)

    # Step 4: apply Gamma(b)^T (vectorized even/odd)
    h = apply_gamma_transpose_batched(self.sorf_b_real, self.sorf_b_imag, h)

    # Step 5: h = h @ S^T
    h = h @ self.sorf_S.T

    result = result + h
    return result


def create_peft_sorf_vera_model(num_labels, args):
    """Create a PEFT VeRA model and monkey-patch it for SORF-VeRA.

    1. Create standard VeRA model
    2. Build SORF matrix S (shared across layers with same d_out)
    3. For each VeRA module: register S buffer, add b/d parameters,
       freeze original lambdas, monkey-patch forward
    4. Ensure only sorf_d_scales, sorf_d_angles, and classifier are trainable
    """
    model = create_peft_model(num_labels, args)

    r = getattr(args, 'r', 256)
    sorf_seed = getattr(args, 'sorf_seed', 42)
    d_initial = getattr(args, 'd_initial', 0.1)

    # Cache SORF matrices by d_out (shared when dims match)
    sorf_cache = {}

    for name, module in model.named_modules():
        if not (hasattr(module, 'vera_A') and hasattr(module, 'out_features')):
            continue

        d_out = module.out_features
        d_in = module.in_features

        # Build/reuse SORF matrix for this d_out
        if d_out not in sorf_cache:
            sorf_cache[d_out] = build_sorf_matrix(d_out, sorf_seed, device='cpu', dtype=torch.float32)
        S = sorf_cache[d_out]

        module.register_buffer('sorf_S', S)

        # Init SORF-VeRA parameters
        init_params = init_sorf_vera_params(d_out, r, init_scale=d_initial)

        # b params: frozen during client training
        module.sorf_b_real = nn.Parameter(
            init_params['b_real'], requires_grad=False
        )
        module.sorf_b_imag = nn.Parameter(
            init_params['b_imag'], requires_grad=False
        )

        # d params: trainable during client training
        module.sorf_d_scale = nn.Parameter(
            init_params['d_scale'], requires_grad=True
        )
        module.sorf_d_angles = nn.Parameter(
            init_params['d_angles'], requires_grad=True
        )

        # Freeze original VeRA lambdas
        if hasattr(module, 'vera_lambda_b'):
            if isinstance(module.vera_lambda_b, nn.Parameter):
                module.vera_lambda_b.requires_grad = False
            elif isinstance(module.vera_lambda_b, nn.ParameterDict):
                for p in module.vera_lambda_b.values():
                    p.requires_grad = False

        if hasattr(module, 'vera_lambda_d'):
            if isinstance(module.vera_lambda_d, nn.Parameter):
                module.vera_lambda_d.requires_grad = False
            elif isinstance(module.vera_lambda_d, nn.ParameterDict):
                for p in module.vera_lambda_d.values():
                    p.requires_grad = False

        # Store r for use in forward
        module.r = r

        # Monkey-patch forward
        module.forward = types.MethodType(_sorf_vera_forward, module)

    # Final pass: only sorf_d_scale, sorf_d_angles, and classifier are trainable
    for param_name, param in model.named_parameters():
        if 'sorf_d_scale' in param_name or 'sorf_d_angles' in param_name:
            param.requires_grad = True
        elif 'classifier' in param_name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    return model


def create_peft_gpt2_model_e2e(args):
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Define LoRA configuration for language modeling task
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # For language modeling
        inference_mode=False,
        r=args.r,  # The dimension of the low-rank update matrices
        lora_alpha=args.lora_alpha,  # The scaling factor for LoRA layers
        lora_dropout=args.lora_dropout,  # Dropout to apply to LoRA layers
        target_modules=["c_attn", "c_proj"],  # Modules to apply LoRA
    )

    # Apply LoRA to the GPT-2 model
    model = get_peft_model(model, lora_config)
    return model


def create_peft_gpt2_model_e2e_ffa(args):
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Define LoRA configuration for language modeling task
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # For language modeling
        inference_mode=False,
        r=args.r,  # The dimension of the low-rank update matrices
        lora_alpha=args.lora_alpha,  # The scaling factor for LoRA layers
        lora_dropout=args.lora_dropout,  # Dropout to apply to LoRA layers
        target_modules=["c_attn", "c_proj"],  # Modules to apply LoRA
    )

    for name, param in model.named_parameters():
        if "lora_A" in name:
            param.requires_grad = False

    # Apply LoRA to the GPT-2 model
    model = get_peft_model(model, lora_config)
    return model
