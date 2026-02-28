# FedEx-LoRA

## Summary
This project implements federated fine-tuning for GLUE sequence classification using FedEx-style aggregation with LoRA and VeRA variants, plus optional SORF-VeRA. It is geared toward multi-client training with configurable aggregation and GPU placement.

Original repository: https://github.com/CERT-Lab/fedex-lora

Note: Experiments were designed for a Pascal GPU cluster and may need tuning for other hardware.

## Project Structure
```
.
├── data_utils.py
├── fed_agg.py
├── fed_train_e2e.py
├── fed_train_glue.py
├── fedex_vera.sh
├── models.py
├── sorf_vera.py
├── test_sorf_vera.py
├── train_eval.py
├── utils.py
├── requirements.txt
├── requirements-lock-py313.txt
├── data/
│   └── your glue dataset
└── assets/
```

## Setup (Conda, Python 3.13)
```bash
conda create -n fedex-vera-py313 python=3.13
conda activate fedex-vera-py313
pip install -r requirements.txt
```

## Model and Dataset
You must import the model weights and the dataset before running experiments. Point `--model` to a local snapshot or rely on the Hugging Face cache for the model, and ensure GLUE data is available via the datasets cache or a local path.

## Example (fed_train_glue)
The SLURM script uses this call:
```bash
python fed_train_glue.py --model=models/models--FacebookAI--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b --task=rte --agg_type=ours_vera --vera --num_clients=3 --r=128 --rounds=51 --lr=1e-3 --local_epochs=5 --parallel_clients --server_gpu 0 --client_gpus 1,2,3
```
This runs FedEx VeRA on the RTE task with three parallel clients on GPUs 1-3 and aggregation on GPU 0.

## fed_train_glue Flags
- `--task`: GLUE task name (default `cola`).
- `--model`: model name or local path (default `roberta-base`).
- `--r`: LoRA/VeRA rank.
- `--lora_alpha`: LoRA/VeRA scaling alpha.
- `--lora_dropout`: LoRA dropout.
- `--vera`: enable VeRA adaptation.
- `--d_initial`: initial VeRA `d` value.
- `--vera_dropout`: VeRA dropout.
- `--projection_prng_key`: VeRA projection PRNG seed.
- `--save_projection`: whether to save VeRA projection.
- `--fan_in_fan_out`: VeRA fan-in/fan-out.
- `--bias`: VeRA bias mode (`none`, `all`, `lora_only`).
- `--modules_to_save`: extra modules to save with VeRA.
- `--init_weights`: whether to init VeRA weights.
- `--layers_to_transform`: VeRA layer indices to transform.
- `--layers_pattern`: VeRA layer name pattern to transform.
- `--rslora`: enable RSLoRA scaling.
- `--rsvera`: enable RSVeRA scaling.
- `--sorf_seed`: seed for SORF matrices.
- `--batch_size`: batch size per client.
- `--agg_type`: aggregation type (`normal`, `ours`, `ours_vera`, `sorf_vera`, `ffa`).
- `--vera_scale`: scale VeRA residue by lr instead of alpha/r.
- `--num_clients`: number of federated clients.
- `--rounds`: number of federated rounds.
- `--local_epochs`: local epochs per client.
- `--warmup_ratio`: scheduler warmup ratio.
- `--max_seq_length`: max token length.
- `--lr`: learning rate.
- `--seed`: random seed.
- `--parallel_clients`: enable multiprocessing clients.
- `--server_gpu`: GPU id for server aggregation.
- `--client_gpus`: comma-separated GPU ids for clients.

Code was refactored and commented by GitHub Copilot.
