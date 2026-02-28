import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, VeraConfig, get_peft_model
from transformers import GPT2LMHeadModel, RobertaForSequenceClassification

from sorf_vera import (
    apply_gamma_transpose_batched,
    build_lambda_d_rot,
    build_sorf_matrix,
    init_sorf_vera_params,
)


def create_peft_model(num_labels, args):
    model = RobertaForSequenceClassification.from_pretrained(
        args.model, num_labels=num_labels
    )

    if getattr(args, "vera", False):
        peft_config = VeraConfig(
            task_type=TaskType.SEQ_CLS,
            r=getattr(args, "r", 256),
            target_modules=["query", "value"],
            projection_prng_key=getattr(args, "projection_prng_key", 0),
            save_projection=getattr(args, "save_projection", True),
            vera_dropout=getattr(args, "vera_dropout", 0.0),
            d_initial=getattr(args, "d_initial", 0.1),
            fan_in_fan_out=getattr(args, "fan_in_fan_out", False),
            bias=getattr(args, "bias", "none"),
            modules_to_save=getattr(args, "modules_to_save", None),
            init_weights=getattr(args, "init_weights", True),
            layers_to_transform=getattr(args, "layers_to_transform", None),
            layers_pattern=getattr(args, "layers_pattern", None),
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

    return get_peft_model(model, peft_config)


def create_peft_FFA_model(num_labels, args):
    model = RobertaForSequenceClassification.from_pretrained(
        args.model, num_labels=num_labels
    )

    if getattr(args, "vera", False):
        peft_config = VeraConfig(
            task_type=TaskType.SEQ_CLS,
            r=getattr(args, "r", 256),
            target_modules=["query", "value"],
            projection_prng_key=getattr(args, "projection_prng_key", 0),
            save_projection=getattr(args, "save_projection", True),
            vera_dropout=getattr(args, "vera_dropout", 0.0),
            d_initial=getattr(args, "d_initial", 0.1),
            fan_in_fan_out=getattr(args, "fan_in_fan_out", False),
            bias=getattr(args, "bias", "none"),
            modules_to_save=getattr(args, "modules_to_save", None),
            init_weights=getattr(args, "init_weights", True),
            layers_to_transform=getattr(args, "layers_to_transform", None),
            layers_pattern=getattr(args, "layers_pattern", None),
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

    for name, param in model.named_parameters():
        if "lora_A" in name:
            param.requires_grad = False

    return model


def _sorf_vera_forward(self, x, *args, **kwargs):
    """Monkey-patched forward for SORF-VeRA modules."""
    result = self.base_layer(x, *args, **kwargs)

    A = self.vera_A["default"]
    B = self.vera_B["default"]

    d_out = self.out_features
    d_in = self.in_features
    r = self.r

    sliced_A = A[:r, :d_in]
    sliced_B = B[:d_out, :r]

    # Keep this on functional ops so the SORF path stays differentiable.
    h = F.linear(x, sliced_A)

    Ld_rot = build_lambda_d_rot(
        self.sorf_d_scale,
        self.sorf_d_angles,
        r,
        device=x.device,
        dtype=x.dtype,
    )
    h = h @ Ld_rot.T

    h = F.linear(h, sliced_B)
    h = apply_gamma_transpose_batched(self.sorf_b_real, self.sorf_b_imag, h)
    h = h @ self.sorf_S.T

    return result + h


def create_peft_sorf_vera_model(num_labels, args):
    """Create a PEFT VeRA model and monkey-patch it for SORF-VeRA."""
    model = create_peft_model(num_labels, args)

    r = getattr(args, "r", 256)
    sorf_seed = getattr(args, "sorf_seed", 42)
    d_initial = getattr(args, "d_initial", 0.1)

    # Cache SORF matrices to reuse mixers across layers with same output size.
    sorf_cache = {}

    for name, module in model.named_modules():
        if not (hasattr(module, "vera_A") and hasattr(module, "out_features")):
            continue

        d_out = module.out_features
        d_in = module.in_features

        if d_out not in sorf_cache:
            sorf_cache[d_out] = build_sorf_matrix(
                d_out, sorf_seed, device="cpu", dtype=torch.float32
            )
        S = sorf_cache[d_out]

        module.register_buffer("sorf_S", S)

        init_params = init_sorf_vera_params(d_out, r, init_scale=d_initial)

        module.sorf_b_real = nn.Parameter(init_params["b_real"], requires_grad=False)
        module.sorf_b_imag = nn.Parameter(init_params["b_imag"], requires_grad=False)

        module.sorf_d_scale = nn.Parameter(init_params["d_scale"], requires_grad=True)
        module.sorf_d_angles = nn.Parameter(
            init_params["d_angles"], requires_grad=True
        )

        if hasattr(module, "vera_lambda_b"):
            if isinstance(module.vera_lambda_b, nn.Parameter):
                module.vera_lambda_b.requires_grad = False
            elif isinstance(module.vera_lambda_b, nn.ParameterDict):
                for p in module.vera_lambda_b.values():
                    p.requires_grad = False

        if hasattr(module, "vera_lambda_d"):
            if isinstance(module.vera_lambda_d, nn.Parameter):
                module.vera_lambda_d.requires_grad = False
            elif isinstance(module.vera_lambda_d, nn.ParameterDict):
                for p in module.vera_lambda_d.values():
                    p.requires_grad = False

        module.r = r
        module.forward = types.MethodType(_sorf_vera_forward, module)

    for param_name, param in model.named_parameters():
        if "sorf_d_scale" in param_name or "sorf_d_angles" in param_name:
            param.requires_grad = True
        elif "classifier" in param_name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    return model


def create_peft_gpt2_model_e2e(args):
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["c_attn", "c_proj"],
    )

    return get_peft_model(model, lora_config)


def create_peft_gpt2_model_e2e_ffa(args):
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["c_attn", "c_proj"],
    )

    for name, param in model.named_parameters():
        if "lora_A" in name:
            param.requires_grad = False

    return get_peft_model(model, lora_config)
