"""Offline evaluation (v2-compatible) for legacy non_neg solo-learn SimCLR checkpoints.

Reads the checkpoint and adjacent args.json from
  trained_models/<method>/<run_id>/
and computes the same metrics as the main project's src/main_eval2.py,
saving output to:
  <ckpt_dir>/eval2_metrics/metrics_{epoch:04d}.npz

Usage:
  uv run python main_eval2.py --ckpt trained_models/simclr/hywyrz38/simclr-resnet18-cifar100-hywyrz38-ep=199.ckpt
"""

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18


# ---------------------------------------------------------------------------
# Normalization — matches pretrain pipeline (build_transform_pipeline cifar100)
# ---------------------------------------------------------------------------

CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD  = (0.2673, 0.2564, 0.2762)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

class _TwoViewDataset(Dataset):
    def __init__(self, base: Dataset, transform: Callable):
        self.base      = base
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        img, label = self.base[idx]
        return self.transform(img), self.transform(img), label


def _build_aug_transform(aug_cfg: dict) -> transforms.Compose:
    """Reconstruct the pretrain augmentation pipeline from args.json augmentation config."""
    import random as _random
    from PIL import ImageFilter

    class _GaussianBlur:
        def __call__(self, img):
            return img.filter(ImageFilter.GaussianBlur(radius=_random.uniform(0.1, 2.0)))

    steps: List = []

    rrc = aug_cfg.get("rrc", {})
    if rrc.get("enabled", False):
        steps.append(transforms.RandomResizedCrop(
            aug_cfg["crop_size"],
            scale=(rrc["crop_min_scale"], rrc["crop_max_scale"]),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ))
    else:
        steps.append(transforms.Resize(aug_cfg["crop_size"]))

    cj = aug_cfg.get("color_jitter", {})
    if cj.get("prob", 0):
        steps.append(transforms.RandomApply(
            [transforms.ColorJitter(cj["brightness"], cj["contrast"], cj["saturation"], cj["hue"])],
            p=cj["prob"],
        ))

    if aug_cfg.get("grayscale", {}).get("prob", 0):
        steps.append(transforms.RandomGrayscale(p=aug_cfg["grayscale"]["prob"]))

    if aug_cfg.get("gaussian_blur", {}).get("prob", 0):
        steps.append(transforms.RandomApply([_GaussianBlur()], p=aug_cfg["gaussian_blur"]["prob"]))

    if aug_cfg.get("horizontal_flip", {}).get("prob", 0):
        steps.append(transforms.RandomHorizontalFlip(p=aug_cfg["horizontal_flip"]["prob"]))

    steps.append(transforms.ToTensor())
    steps.append(transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD))
    return transforms.Compose(steps)


def _cifar100_loader(
    data_root: str,
    train: bool,
    batch_size: int,
    num_workers: int,
    transform: Callable,
    two_view: bool = False,
) -> DataLoader:
    pin = torch.cuda.is_available()
    if two_view:
        base = torchvision.datasets.CIFAR100(data_root, train=True, download=False, transform=None)
        ds: Dataset = _TwoViewDataset(base, transform)
        return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True,
                          num_workers=num_workers, pin_memory=pin,
                          persistent_workers=num_workers > 0)
    ds = torchvision.datasets.CIFAR100(data_root, train=train, download=False, transform=transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=train, drop_last=train,
                      num_workers=num_workers, pin_memory=pin,
                      persistent_workers=num_workers > 0)


# ---------------------------------------------------------------------------
# NTXent loss (inline — avoids cross-project imports)
# ---------------------------------------------------------------------------

class NTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        B = z1.size(0)
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        z   = torch.cat([z1, z2], dim=0)
        sim = torch.mm(z, z.T) / self.temperature
        sim = sim.masked_fill(torch.eye(2 * B, dtype=torch.bool, device=z.device), float("-inf"))
        labels = torch.cat([torch.arange(B, 2 * B), torch.arange(B)]).to(z.device)
        return F.cross_entropy(sim, labels)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class _Exp(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(x)


_NON_NEG_MODULES: Dict[str, nn.Module] = {
    "relu":      nn.ReLU(),
    "rep_relu":  nn.ReLU(),      # forward is ReLU; GELU backward only matters during training
    "gelu":      nn.GELU(),
    "sigmoid":   nn.Sigmoid(),
    "softplus":  nn.Softplus(),
    "leakyrelu": nn.LeakyReLU(),
    "exp":       _Exp(),
}


class LegacySimCLR(nn.Module):
    """Drop-in adapter with the same interface as src/model.py SimCLRModel."""

    def __init__(self, backbone: nn.Module, projector: nn.Sequential):
        super().__init__()
        self.backbone  = backbone
        self.projector = projector

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).flatten(1)


def load_legacy_checkpoint(ckpt_path: Path, run_args: dict) -> Tuple[LegacySimCLR, int]:
    """Construct and load a LegacySimCLR from a PyTorch-Lightning checkpoint.

    The non_neg activation (if any) is appended as an extra layer in projector so
    that projector[:2] still returns the 2048-dim hidden representation and
    projector(enc) returns the activation-gated output — matching the training setup.
    """
    mk = run_args["method_kwargs"]
    proj_hidden = mk["proj_hidden_dim"]
    proj_output = mk["proj_output_dim"]
    non_neg     = mk.get("non_neg")

    # Backbone — ResNet18 with the legacy CIFAR patch used in solo-learn base.py:
    # 3×3 conv with padding=2, no maxpool, fc removed
    backbone = resnet18(weights=None)
    backbone.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
    backbone.maxpool  = nn.Identity()
    backbone.fc       = nn.Identity()

    # Projector — 2-layer MLP (no BN), matching legacy solo SimCLR
    proj_layers: List[nn.Module] = [
        nn.Linear(512, proj_hidden),
        nn.ReLU(),
        nn.Linear(proj_hidden, proj_output),
    ]
    if non_neg is not None and non_neg in _NON_NEG_MODULES:
        proj_layers.append(_NON_NEG_MODULES[non_neg])
    projector = nn.Sequential(*proj_layers)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd   = ckpt["state_dict"]

    backbone_sd  = {k[len("backbone."):]:  v for k, v in sd.items() if k.startswith("backbone.")}
    projector_sd = {k[len("projector."):]: v for k, v in sd.items() if k.startswith("projector.")}

    backbone.load_state_dict(backbone_sd, strict=True)
    # Activation layers have no parameters, so strict=True works even with the extra trailing layer
    projector.load_state_dict(projector_sd, strict=True)

    return LegacySimCLR(backbone, projector), int(ckpt["epoch"])


# ---------------------------------------------------------------------------
# Feature extraction — tensors stay on device throughout
# ---------------------------------------------------------------------------

@torch.no_grad()
def _extract_val_features(
    model: LegacySimCLR,
    loader: DataLoader,
    device: torch.device,
    hidden_slice: int,
) -> Dict[str, torch.Tensor]:
    nb = device.type == "cuda"
    enc_list, hidden_list, proj_list, label_list = [], [], [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=nb)
        enc    = model.encode(images)
        enc_list.append(enc)
        hidden_list.append(model.projector[:hidden_slice](enc))
        proj_list.append(model.projector(enc))
        label_list.append(labels.to(device, non_blocking=nb))
    return {
        "encoder":     torch.cat(enc_list),
        "proj_hidden": torch.cat(hidden_list),
        "proj":        torch.cat(proj_list),
        "labels":      torch.cat(label_list),
    }


@torch.no_grad()
def _extract_with_labels(
    encode_fn: Callable[[torch.Tensor], torch.Tensor],
    loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    nb = device.type == "cuda"
    feat_list, label_list = [], []
    for images, labels in loader:
        feat_list.append(encode_fn(images.to(device, non_blocking=nb)))
        label_list.append(labels.to(device, non_blocking=nb))
    return torch.cat(feat_list), torch.cat(label_list)


# ---------------------------------------------------------------------------
# Sparsity metrics
# ---------------------------------------------------------------------------

def sparsity(features: torch.Tensor) -> Dict[str, float]:
    N, D = features.shape
    l1 = features.abs().sum(dim=1)
    l2 = features.norm(dim=1)
    hoyer    = ((D ** 0.5 - l1 / l2.clamp(min=1e-8)) / (D ** 0.5 - 1)).mean()
    zero_pct = (features.abs() < 1e-5).float().mean()
    return {"hoyer": hoyer.item(), "zero_pct": zero_pct.item()}


# ---------------------------------------------------------------------------
# Class consistency
# ---------------------------------------------------------------------------

def class_consistency(
    features: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 1e-5,
) -> Dict[str, float]:
    N, D        = features.shape
    num_classes = int(labels.max().item()) + 1
    active      = features > threshold                                   # (N, D)
    n_active    = active.sum(dim=0)                                      # (D,)
    one_hot     = torch.zeros(N, num_classes, device=features.device)
    one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
    count_per_class = one_hot.T @ active.float()                         # (C, D)
    most_frequent   = count_per_class.max(dim=0).values
    rates           = most_frequent / n_active.float().clamp(min=1)
    return {"class_consistency": rates.mean().item()}


# ---------------------------------------------------------------------------
# Dimensional correlation
# ---------------------------------------------------------------------------

def dim_correlation(features: torch.Tensor, n_dims: int = 20, seed: int = 0) -> Dict:
    N, D = features.shape
    idx  = np.random.default_rng(seed).choice(D, min(n_dims, D), replace=False)
    idx_t  = torch.from_numpy(idx).long().to(features.device)
    F_sub  = features[:, idx_t]
    F_norm = F_sub / F_sub.norm(dim=0).clamp(min=1e-8)
    C      = F_norm.T @ F_norm
    mask   = ~torch.eye(C.shape[0], dtype=torch.bool, device=C.device)
    return {
        "dim_corr_matrix":       C.cpu().numpy(),
        "dim_corr_indices":      idx,
        "dim_corr_mean_offdiag": C[mask].abs().mean().item(),
    }


# ---------------------------------------------------------------------------
# Expected Activation
# ---------------------------------------------------------------------------

def compute_ea(train_features: torch.Tensor) -> torch.Tensor:
    norms     = train_features.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return (train_features / norms).mean(dim=0)


# ---------------------------------------------------------------------------
# Linear probe — fully on-device, no DataLoader overhead
# ---------------------------------------------------------------------------

def _train_linear_probe(
    train_feats:  torch.Tensor,
    train_labels: torch.Tensor,
    num_classes:  int,
    epochs:       int = 100,
    lr:           float = 0.1,
    batch_size:   int = 1024,
) -> nn.Linear:
    N      = train_feats.shape[0]
    device = train_feats.device
    clf    = nn.Linear(train_feats.shape[1], num_classes, device=device)
    opt    = torch.optim.SGD(clf.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    clf.train()
    for _ in range(epochs):
        perm = torch.randperm(N, device=device)
        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            F.cross_entropy(clf(train_feats[idx]), train_labels[idx]).backward()
            opt.step()
            opt.zero_grad()
    return clf


@torch.no_grad()
def _eval_accuracy(clf: nn.Linear, feats: torch.Tensor, labels: torch.Tensor) -> float:
    clf.eval()
    return (clf(feats).argmax(dim=1) == labels).float().mean().item()


@torch.no_grad()
def _eval_map_at_k(feats: torch.Tensor, labels: torch.Tensor, k: int = 10) -> float:
    N       = feats.shape[0]
    feats_n = F.normalize(feats.float(), dim=1)
    arange_k = torch.arange(1, k + 1, dtype=torch.float32, device=feats.device)
    # pre-compute number of relevant items per query (excluding self)
    n_rel_all = torch.bincount(labels, minlength=int(labels.max().item()) + 1)[labels] - 1  # (N,)

    ap_sum = 0.0
    chunk  = 512
    for start in range(0, N, chunk):
        end  = min(start + chunk, N)
        B    = end - start
        sims = feats_n[start:end] @ feats_n.T                           # (B, N)

        diag = torch.arange(B, device=feats.device)
        sims[diag, start + diag] = float("-inf")

        top_idx    = sims.topk(k, dim=1).indices                        # (B, k)
        top_labels = labels[top_idx]                                     # (B, k)
        q_labels   = labels[start:end]                                   # (B,)

        hits  = (top_labels == q_labels.unsqueeze(1)).float()           # (B, k)
        n_rel = n_rel_all[start:end]                                     # (B,)
        denom = n_rel.clamp(max=k, min=1).float()

        prec  = hits.cumsum(dim=1) / arange_k                           # (B, k)
        ap    = (prec * hits).sum(dim=1) / denom                        # (B,)
        ap_sum += ap[n_rel > 0].sum().item()

    return ap_sum / N


# ---------------------------------------------------------------------------
# Feature-subset evaluation
# ---------------------------------------------------------------------------

def evaluate_feature_subsets(
    encode_fn:    Callable[[torch.Tensor], torch.Tensor],
    train_loader: DataLoader,
    val_loader:   DataLoader,
    num_classes:  int,
    device:       torch.device,
    n_select:     int = 128,
    probe_epochs: int = 100,
    probe_lr:     float = 0.1,
    seed:         int = 0,
) -> Dict:
    """Linear probe accuracy and mAP@10 for four feature subsets (all / rand / ea / ea+relu)."""
    print("  extracting train features...", end=" ", flush=True)
    train_feats, train_labels = _extract_with_labels(encode_fn, train_loader, device)
    print("done")
    print("  extracting val features...", end=" ", flush=True)
    val_feats, val_labels = _extract_with_labels(encode_fn, val_loader, device)
    print("done")

    D   = train_feats.shape[1]
    ea  = compute_ea(train_feats)

    rng      = np.random.default_rng(seed)
    rand_idx = torch.from_numpy(rng.choice(D, n_select, replace=False)).long().to(device)
    top_idx  = ea.argsort()[-n_select:]

    n = n_select
    subsets = [
        ("all",        torch.arange(D, device=device), False),
        (f"rand{n}",   rand_idx,                        False),
        (f"ea{n}",     top_idx,                         False),
        (f"ea{n}relu", top_idx,                         True),
    ]

    results: Dict = {}
    for name, idx, use_relu in subsets:
        tr = train_feats[:, idx]
        vl = val_feats[:, idx]
        if use_relu:
            tr = tr.clamp(min=0)
            vl = vl.clamp(min=0)

        print(f"  [{name:12s}] training probe...", end=" ", flush=True)
        clf   = _train_linear_probe(tr, train_labels, num_classes,
                                    epochs=probe_epochs, lr=probe_lr)
        print("done", end="  ")

        acc   = _eval_accuracy(clf, vl, val_labels)
        map10 = _eval_map_at_k(vl, val_labels, k=10)
        print(f"acc={acc*100:.2f}%  mAP@10={map10*100:.2f}%")

        results[f"{name}_acc"]   = acc
        results[f"{name}_map10"] = map10

    return results


# ---------------------------------------------------------------------------
# SEPIN / disentanglement
# ---------------------------------------------------------------------------

@torch.no_grad()
def _batched_nce(
    z1: torch.Tensor, z2: torch.Tensor,
    loss_fn: NTXentLoss, batch_size: int,
) -> float:
    """NTXent averaged over mini-batches. z1/z2 must already be on the target device."""
    total, n_batches = 0.0, 0
    for start in range(0, len(z1), batch_size):
        total += loss_fn(z1[start:start + batch_size], z2[start:start + batch_size]).item()
        n_batches += 1
    return total / n_batches


@torch.no_grad()
def compute_disentanglement(
    encode_fn:             Callable[[torch.Tensor], torch.Tensor],
    train_loader_two_view: DataLoader,
    device:                torch.device,
    temperature:           float = 0.1,
    nce_batch_size:        int   = 256,
) -> Dict:
    """SEPIN@1/10/100/all via leave-one-out NTXent on two-view train features."""
    nb = device.type == "cuda"
    print("  extracting two-view train features...", end=" ", flush=True)
    z1_list, z2_list = [], []
    for x1, x2, _ in train_loader_two_view:
        z1_list.append(encode_fn(x1.to(device, non_blocking=nb)))
        z2_list.append(encode_fn(x2.to(device, non_blocking=nb)))
    z1 = torch.cat(z1_list)   # (N, D) — stays on device
    z2 = torch.cat(z2_list)
    print("done")

    D       = z1.shape[1]
    loss_fn = NTXentLoss(temperature)

    nce_full = _batched_nce(z1, z2, loss_fn, nce_batch_size)
    print(f"  NTXent full (D={D}): {nce_full:.4f}")

    # Leave-one-out: index mask built on device, all slicing stays on device
    all_idx = torch.arange(D, device=device)
    deltas  = torch.empty(D)
    for i in range(D):
        keep      = torch.cat([all_idx[:i], all_idx[i + 1:]])
        deltas[i] = _batched_nce(z1[:, keep], z2[:, keep], loss_fn, nce_batch_size) - nce_full

    deltas_ranked = deltas.sort(descending=True).values

    results: Dict = {"nce_full": nce_full}
    for k in (1, 10, 100, D):
        label = "all" if k == D else str(k)
        results[f"sepin_{label}"] = float(deltas_ranked[:k].mean().item())

    print(f"  SEPIN@1={results['sepin_1']:.4f}  @10={results['sepin_10']:.4f}"
          f"  @100={results['sepin_100']:.4f}  @all={results['sepin_all']:.4f}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Offline evaluation v2 for legacy non_neg solo SimCLR checkpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt", required=True,
                   help="Path to .ckpt file (args.json must be in the same directory).")
    p.add_argument("--data-root", default=None,
                   help="Override CIFAR-100 data root from args.json.")
    p.add_argument("--num-workers", default=None, type=int,
                   help="Override num_workers from args.json.")
    p.add_argument("--n-select", default=128, type=int,
                   help="Feature dims for rand/EA subset evaluations.")
    p.add_argument("--probe-epochs", default=100, type=int)
    p.add_argument("--probe-lr", default=0.1, type=float)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    args_path = ckpt_path.parent / "args.json"
    if not args_path.exists():
        raise SystemExit(f"args.json not found alongside checkpoint: {args_path}")

    run_args = json.loads(args_path.read_text())

    data_root   = args.data_root   or run_args["data"]["train_path"]
    num_workers = args.num_workers if args.num_workers is not None else run_args["data"]["num_workers"]
    batch_size  = run_args["optimizer"]["batch_size"]
    temperature = run_args["method_kwargs"].get("temperature", 0.2)
    num_classes = run_args["data"]["num_classes"]
    non_neg     = run_args["method_kwargs"].get("non_neg")

    print(f"Checkpoint : {ckpt_path}")
    print(f"Model name : {run_args['name']}")
    print(f"non_neg    : {non_neg}")
    print(f"Data root  : {data_root}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")

    # --- Data ---
    clean_t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    aug_t = _build_aug_transform(run_args["augmentations"][0])

    val_loader      = _cifar100_loader(data_root, train=False, batch_size=batch_size,
                                       num_workers=num_workers, transform=clean_t)
    train_loader    = _cifar100_loader(data_root, train=True,  batch_size=batch_size,
                                       num_workers=num_workers, transform=clean_t)
    train_loader_tv = _cifar100_loader(data_root, train=True,  batch_size=batch_size,
                                       num_workers=num_workers, transform=aug_t, two_view=True)

    # --- Model ---
    model, epoch = load_legacy_checkpoint(ckpt_path, run_args)
    model.to(device).eval()
    print(f"Loaded epoch {epoch}")

    # Legacy SimCLR projector is always a plain 2-layer MLP (type "mlp"), hidden_slice=2.
    # For non_neg models, the activation is appended as projector[3], so [:2] still gives the
    # 2048-dim hidden representation — identical to the main project's convention.
    hidden_slice = 2

    # --- Val feature extraction ---
    print("\nExtracting val features...")
    features   = _extract_val_features(model, val_loader, device, hidden_slice)
    val_labels = features["labels"].long()

    layers = [
        ("encoder",     "encoder",
         lambda imgs: model.encode(imgs)),
        ("proj_hidden", "proj_hidden",
         lambda imgs, s=hidden_slice: model.projector[:s](model.encode(imgs))),
        ("proj",        "proj",
         lambda imgs: model.projector(model.encode(imgs))),
    ]

    results: Dict = {"epoch": epoch}

    for name, feat_key, encode_fn in layers:
        val_feats = features[feat_key]

        m = {
            **sparsity(val_feats),
            **class_consistency(val_feats, val_labels),
            **dim_correlation(val_feats),
        }
        results.update({f"{name}_{k}": v for k, v in m.items()})

        print(f"\n--- Feature subsets: {name} ---")
        sub = evaluate_feature_subsets(
            encode_fn, train_loader, val_loader, num_classes, device,
            n_select=args.n_select, probe_epochs=args.probe_epochs, probe_lr=args.probe_lr,
        )
        results.update({f"{name}_{k}": v for k, v in sub.items()})

        print(f"\n--- Disentanglement: {name} ---")
        dis = compute_disentanglement(
            encode_fn, train_loader_tv, device,
            temperature=temperature, nce_batch_size=batch_size,
        )
        results.update({f"{name}_{k}": v for k, v in dis.items()})

    # --- Print summary ---
    col_w = max(len(k) for k in results) + 2
    print(f"\n{'Metric':<{col_w}} Value")
    print("-" * (col_w + 10))
    for k, v in results.items():
        if isinstance(v, (np.ndarray, torch.Tensor)):
            continue
        print(f"{k:<{col_w}} {v:.4f}" if isinstance(v, float) else f"{k:<{col_w}} {v}")

    # --- Save metrics ---
    def _to_numpy(v) -> np.ndarray:
        if isinstance(v, torch.Tensor):
            return v.cpu().numpy()
        return np.asarray(v)

    metrics_path = ckpt_path.parent / "eval2_metrics" / f"metrics_{epoch:04d}.npz"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(metrics_path, **{k: _to_numpy(v) for k, v in results.items()})  # type: ignore[call-overload]
    print(f"\nSaved metrics → {metrics_path}")


if __name__ == "__main__":
    main()
