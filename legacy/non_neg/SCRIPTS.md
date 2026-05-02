# Scripts

All scripts are run with `uv run python <script>` from this directory.

---

## Dataset preparation

### `create_imagenet100.py`
Extracts the ImageNet-100 subset from ILSVRC2012 tar files.
Selectively unpacks only the 100 needed classes — does not require extracting the full 137 GB.
Requires ~13 GB of free disk space.

```bash
uv run python create_imagenet100.py \
  --dataset-path /path/to/ilsvrc2012 \
  --output       ./imagenet100
```

Expected files in `--dataset-path`:
- `ILSVRC2012_img_train.tar`
- `ILSVRC2012_img_val.tar`
- `ILSVRC2012_devkit_t12.tar.gz`

CIFAR-100 is downloaded automatically by torchvision on first use.

---

## Config

### `make_args_json.py`
Converts a Hydra launch YAML into an `args.json` compatible with the eval scripts.
Resolves augmentation/wandb sub-configs and fills in all training defaults.

```bash
# baseline SimCLR
uv run python make_args_json.py \
  scripts/pretrain/cifar/simclr.yaml \
  --output trained_models/simclr/myrun

# with overrides (Hydra dotlist syntax)
uv run python make_args_json.py \
  scripts/pretrain/cifar/simclr.yaml \
  --output trained_models/simclr/myrun \
  method_kwargs.non_neg=rep_relu \
  name=simclr-resnet18-cifar100-ncl
```

---

## Training

### `main_pretrain.py`
Pretrains a SimCLR model using the solo-learn framework and PyTorch Lightning.
Config is a Hydra YAML; hyperparameters can be overridden on the command line.

```bash
# CIFAR-100 baseline
uv run python main_pretrain.py \
  --config-path scripts/pretrain/cifar \
  --config-name simclr.yaml

# CIFAR-100 with rep_relu
uv run python main_pretrain.py \
  --config-path scripts/pretrain/cifar \
  --config-name simclr.yaml \
  method_kwargs.non_neg=rep_relu \
  name=simclr-resnet18-cifar100-ncl

# ImageNet-100
uv run python main_pretrain.py \
  --config-path scripts/pretrain/imagenet-100 \
  --config-name simclr.yaml \
  data.train_path=./imagenet100/train \
  data.val_path=./imagenet100/val
```

Checkpoints are saved to `trained_models/<method>/<wandb_run_id>/`.

### `main_linear.py`
Trains a linear classifier on top of a frozen pretrained backbone (linear probing).

```bash
uv run python main_linear.py \
  --config-path scripts/linear/cifar \
  --config-name simclr.yaml \
  pretrained_feature_extractor=trained_models/simclr/hywyrz38/simclr-resnet18-cifar100-hywyrz38-ep=199.ckpt
```

---

## Evaluation

### `main_eval2.py`
Full offline evaluation of a trained checkpoint. Computes sparsity, class consistency,
dimensional correlation, linear probe accuracy, mAP@10, and SEPIN disentanglement
for the encoder, proj_hidden, and proj layers. Saves results to
`<run_dir>/eval2_metrics/metrics_{epoch:04d}.npz`.

Supports CIFAR-100 and ImageNet-100 (detected automatically from `args.json`).

```bash
uv run python main_eval2.py --path trained_models/simclr/hywyrz38

# specific epoch
uv run python main_eval2.py --path trained_models/simclr/hywyrz38 --epoch 100

# override data path
uv run python main_eval2.py --path trained_models/simclr/hywyrz38 --data-root ./data
```

### `main_disent.py`
Computes only the SEPIN disentanglement metrics for a single layer. Faster than
running the full `main_eval2.py` when only disentanglement values are needed.

```bash
uv run python main_disent.py \
  --path  trained_models/simclr/hywyrz38 \
  --layer proj

# all layer choices: encoder, proj_hidden, proj
```

### `main_eval.py`
Legacy evaluation script (exploratory, not maintained). Use `main_eval2.py` instead.

---

## Typical workflow

```bash
# 1. Pretrain
uv run python main_pretrain.py --config-path scripts/pretrain/cifar --config-name simclr.yaml

# 2. Evaluate (full)
uv run python main_eval2.py --path trained_models/simclr/<run_id>

# 3. Quick disentanglement check during development
uv run python main_disent.py --path trained_models/simclr/<run_id> --layer proj
```
