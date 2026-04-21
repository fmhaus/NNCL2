"""Offline evaluation for a saved SimCLR run.

Reads model config from saves/<name>/hparams.json so only --name is required.
Runs eval_layer for encoder output and projected features, then kNN and linear
probe accuracy. Results are printed to console and appended to a separate
eval_metrics.csv in the run directory.
"""

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from dataset import load_dataset
from logger import TrainingLogger
from model import SimCLRModel, LinearClassifier
from main_train import _topk_accuracy, _wrap_tqdm, knn_accuracy


# ---------------------------------------------------------------------------
# eval_layer
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_layer(
    layer_fn: Callable[[torch.Tensor], torch.Tensor],
    loader,
    name: str,
    logger: TrainingLogger,
    epoch: int,
    device: torch.device,
) -> dict:
    """Evaluate a layer's activations over the full val set.

    Scalar statistics (L1, L2, Hoyer) are aggregated over all batches for
    reliable estimates. The activation histogram is saved from the first batch
    only — the distribution shape is well-characterised by a few hundred
    samples, and collecting all activations would be memory-intensive.

    Args:
        layer_fn: maps a batch of images (already on device) → (B, D) tensor
        loader:   eval DataLoader yielding (images, labels) pairs
        name:     key prefix for returned metrics and histogram filename
        logger:   TrainingLogger used to save the histogram .npz
        epoch:    checkpoint epoch, used to label the saved histogram
        device:   torch device

    Returns:
        dict with keys ``{name}_l1``, ``{name}_l2``, ``{name}_hoyer``
    """
    is_cuda = device.type == "cuda"
    sum_l1 = sum_l2 = sum_hoyer = n = 0
    first = True

    for images, _ in loader:
        images = images.to(device, non_blocking=is_cuda)
        acts   = layer_fn(images)   # (B, D)

        if first:
            # Single-batch histogram: sufficient to characterise the shape
            logger.log_activation_histogram(epoch, name, acts)
            first = False

        b  = acts.size(0)
        d  = acts.size(1)
        l1 = acts.norm(p=1, dim=1)
        l2 = acts.norm(p=2, dim=1)
        sum_l1    += l1.mean().item() * b
        sum_l2    += l2.mean().item() * b
        sum_hoyer += ((d ** 0.5 - l1 / l2.clamp(min=1e-8)) / (d ** 0.5 - 1)).mean().item() * b
        n += b

    return {
        f"{name}_l1":    sum_l1    / n,
        f"{name}_l2":    sum_l2    / n,
        f"{name}_hoyer": sum_hoyer / n,
    }


# ---------------------------------------------------------------------------
# Disentanglement / completeness from an arbitrary R matrix
# ---------------------------------------------------------------------------

def disentanglement(R: np.ndarray) -> tuple[np.ndarray, float]:
    """Disentanglement score per feature dimension and as a weighted scalar.

    Args:
        R: (C, D) relative-importance matrix, non-negative.

    Returns:
        per_dim: (D,) score in [0, 1]; 1 = dimension important for exactly one class
        scalar:  weighted average, weighted by each dimension's total importance
    """
    C = R.shape[0]
    col_sum = R.sum(axis=0).clip(min=1e-8)                   # (D,)
    p = R / col_sum[None, :]                                  # (C, D)
    log_p = np.where(p > 0, np.log(p), 0.0)
    h = -(p * log_p).sum(axis=0) / np.log(C)                 # (D,) normalised entropy
    per_dim = 1.0 - h                                         # (D,)
    weights = col_sum / col_sum.sum()                         # (D,) relative importance of each dim
    return per_dim, float((weights * per_dim).sum())


def completeness(R: np.ndarray) -> tuple[np.ndarray, float]:
    """Completeness score per class and as a weighted scalar.

    Args:
        R: (C, D) relative-importance matrix, non-negative.

    Returns:
        per_class: (C,) score in [0, 1]; 1 = class explained by exactly one dimension
        scalar:    weighted average, weighted by each class's total importance
    """
    D = R.shape[1]
    row_sum = R.sum(axis=1).clip(min=1e-8)                   # (C,)
    p = R / row_sum[:, None]                                  # (C, D)
    log_p = np.where(p > 0, np.log(p), 0.0)
    h = -(p * log_p).sum(axis=1) / np.log(D)                 # (C,)
    per_class = 1.0 - h                                       # (C,)
    weights = row_sum / row_sum.sum()                         # (C,)
    return per_class, float((weights * per_class).sum())


# ---------------------------------------------------------------------------
# Per-class feature statistics
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_per_class_stats(
    layer_fn: Callable[[torch.Tensor], torch.Tensor],
    loader,
    num_classes: int,
    save_path: Path,
    device: torch.device,
) -> None:
    """Compute per-class feature statistics and representation quality metrics.

    Accumulates online sufficient statistics (sum, sum-of-squares, count) so
    the full feature matrix never needs to be held in memory.

    Saved arrays in ``save_path``:
        mean              (C, D)  per-class mean
        var               (C, D)  per-class variance
        within_var        (D,)    weighted mean of per-class variances
        between_var       (D,)    variance of class means weighted by class counts
        separability      (D,)    between_var / total_var ∈ [0, 1]
        mean_separability scalar  mean separability across all dimensions
        R                 (C, D)  relative importance: each class's contribution to separability
        disentanglement   (D,)    per-dimension disentanglement score ∈ [0, 1]
        completeness      (C,)    per-class completeness score ∈ [0, 1]
    """
    is_cuda = device.type == "cuda"

    feat_sum:    torch.Tensor | None = None
    feat_sum_sq: torch.Tensor | None = None
    counts:      torch.Tensor | None = None
    sum_l1 = sum_l2 = sum_hoyer = sum_zero = n_total = 0.0
    d = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=is_cuda)
        labels = labels.to(device, non_blocking=is_cuda)
        acts   = layer_fn(images)   # (B, D)

        if feat_sum is None:
            d = acts.size(1)
            feat_sum    = torch.zeros(num_classes, d, device=device)
            feat_sum_sq = torch.zeros(num_classes, d, device=device)
            counts      = torch.zeros(num_classes,    device=device)
        assert feat_sum is not None and feat_sum_sq is not None and counts is not None

        feat_sum.scatter_add_(0, labels.unsqueeze(1).expand_as(acts), acts)
        feat_sum_sq.scatter_add_(0, labels.unsqueeze(1).expand_as(acts), acts ** 2)
        counts.scatter_add_(0, labels, torch.ones(labels.size(0), device=device))

        b   = acts.size(0)
        l1  = acts.norm(p=1, dim=1)
        l2  = acts.norm(p=2, dim=1)
        sum_l1    += l1.mean().item() * b
        sum_l2    += l2.mean().item() * b
        sum_hoyer += ((d ** 0.5 - l1 / l2.clamp(min=1e-8)) / (d ** 0.5 - 1)).mean().item() * b
        sum_zero  += (acts == 0).float().mean().item() * b
        n_total   += b

    assert feat_sum is not None and feat_sum_sq is not None and counts is not None
    n  = counts.sum()
    w  = counts / n                                                  # (C,) class weights

    mean        = feat_sum    / counts.unsqueeze(1)                  # (C, D)
    var         = (feat_sum_sq / counts.unsqueeze(1) - mean ** 2).clamp(min=0)  # (C, D)
    global_mean = (w.unsqueeze(1) * mean).sum(0)                     # (D,)
    within_var  = (w.unsqueeze(1) * var).sum(0)                      # (D,)
    between_var = (w.unsqueeze(1) * (mean - global_mean) ** 2).sum(0)  # (D,)
    total_var   = within_var + between_var
    separability = between_var / total_var.clamp(min=1e-8)           # (D,)

    # R_cd = class c's contribution to the separability of dimension d
    # Columns sum to separability by construction
    R_np = (w.unsqueeze(1) * (mean - global_mean) ** 2 / total_var.clamp(min=1e-8)).cpu().numpy()  # (C, D)

    # Dead dimensions: RMS activation < threshold across the full dataset
    # E[x²] = total_var + global_mean²  (second moment = variance + mean²)
    dim_rms        = (total_var + global_mean ** 2).clamp(min=0).sqrt()   # (D,)
    dead_threshold = 1e-3
    dead_dims      = dim_rms < dead_threshold                             # (D,) bool
    dead_fraction  = dead_dims.float().mean().item()

    dis_per_dim, dis_scalar = disentanglement(R_np)
    com_per_class, com_scalar = completeness(R_np)

    save_path.parent.mkdir(exist_ok=True)
    np.savez_compressed(
        save_path,
        mean=mean.cpu().numpy(),
        var=var.cpu().numpy(),
        within_var=within_var.cpu().numpy(),
        between_var=between_var.cpu().numpy(),
        separability=separability.cpu().numpy(),
        mean_separability=separability.mean().item(),
        dim_rms=dim_rms.cpu().numpy(),
        dead_dim_fraction=dead_fraction,
        mean_l1=sum_l1 / n_total,
        mean_l2=sum_l2 / n_total,
        mean_hoyer=sum_hoyer / n_total,
        zero_fraction=sum_zero / n_total,
        R=R_np,
        disentanglement=dis_per_dim,
        disentanglement_score=dis_scalar,
        completeness=com_per_class,
        completeness_score=com_scalar,
    )
    print(f"Saved per-class stats → {save_path}")


# ---------------------------------------------------------------------------
# Linear probe accuracy (no training — weights loaded from checkpoint)
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_linear_probe(model, classifier, loader, device, use_tqdm: bool = False) -> dict:
    model.eval()
    classifier.eval()
    is_cuda = device.type == "cuda"

    sum_loss = sum_acc1 = sum_acc5 = n = 0
    for images, labels in _wrap_tqdm(loader, use_tqdm, desc="linear probe", leave=False):
        images = images.to(device, non_blocking=is_cuda)
        labels = labels.to(device, non_blocking=is_cuda)
        logits = classifier(model.encode(images))
        sum_loss += F.cross_entropy(logits, labels).item()
        acc1, acc5 = _topk_accuracy(logits, labels)
        b = labels.size(0)
        sum_acc1 += acc1 * b
        sum_acc5 += acc5 * b
        n += b

    return {
        "probe_loss": sum_loss / len(loader),
        "probe_acc1": sum_acc1 / n,
        "probe_acc5": sum_acc5 / n,
    }


# ---------------------------------------------------------------------------
# Checkpoint loading (model + classifier weights only)
# ---------------------------------------------------------------------------

def load_eval_checkpoint(path: Path, model: SimCLRModel, classifier: LinearClassifier) -> int:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    if "classifier" in ckpt:
        classifier.load_state_dict(ckpt["classifier"])
    return ckpt["epoch"]


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Offline evaluation for a saved SimCLR run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--name",        required=True,
                   help="Run name — points to saves/<name>/.")
    p.add_argument("--epoch",       default=None, type=int,
                   help="Checkpoint epoch to evaluate. Default: latest.")
    # Environment overrides — everything else is read from hparams.json
    p.add_argument("--data-root",   default=None,
                   help="Override data root from hparams (useful on a different machine).")
    p.add_argument("--num-workers", default=None, type=int,
                   help="Override num_workers from hparams.")
    p.add_argument("--tqdm",        action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args     = parse_args()
    save_dir = Path("saves") / args.name

    if not save_dir.exists():
        raise SystemExit(f"Run directory '{save_dir}' does not exist.")

    # --- Load config from hparams.json ---
    hparams = json.loads((save_dir / "hparams.json").read_text())

    # Environment overrides
    if args.data_root   is not None: hparams["data_root"]   = args.data_root
    if args.num_workers is not None: hparams["num_workers"]  = args.num_workers

    # --- Select checkpoint ---
    ckpts = sorted(save_dir.glob("state_*.ckpt"))
    if not ckpts:
        raise SystemExit(f"No checkpoints found in '{save_dir}'.")

    if args.epoch is not None:
        ckpt_path = save_dir / f"state_{args.epoch:04d}.ckpt"
        if not ckpt_path.exists():
            raise SystemExit(f"Checkpoint '{ckpt_path}' not found.")
    else:
        ckpt_path = ckpts[-1]

    print(f"Evaluating '{args.name}' from {ckpt_path.name}")

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_cuda = device.type == "cuda"

    # --- Data ---
    dataset     = hparams["dataset"]
    image_size  = 32  if dataset == "cifar100" else 64
    num_classes = 100 if dataset == "cifar100" else 200

    def get_loader(train: bool):
        return load_dataset(
            dataset, two_view=False, augment=None, train=train,
            batch_size=hparams["batch_size"],
            num_workers=hparams["num_workers"],
            data_root=hparams["data_root"],
        )

    knn_train = get_loader(train=True)
    val_loader = get_loader(train=False)

    # --- Model ---
    model = SimCLRModel(
        proj_hidden=hparams["proj_hidden_dim"],
        proj_dim=hparams["proj_output_dim"],
        image_size=image_size,
        projector=hparams.get("projector", "mlp"),
    ).to(device)
    classifier = LinearClassifier(num_classes=num_classes).to(device)

    epoch = load_eval_checkpoint(ckpt_path, model, classifier)
    model.eval()
    classifier.eval()

    # logger writes histograms to saves/<name>/histograms/
    logger = TrainingLogger(save_dir, argparse.Namespace(**hparams), console_log=False)

    projector_type = hparams.get("projector", "mlp")
    per_class_dir  = save_dir / "per_class_stats"

    # --- Layer evaluations ---
    def _run_layer(name: str, layer_fn: Callable[[torch.Tensor], torch.Tensor]) -> dict:
        stats = eval_layer(layer_fn, val_loader, name, logger, epoch, device)
        compute_per_class_stats(layer_fn, val_loader, num_classes,
                                per_class_dir / f"{name}_{epoch:04d}.npz", device)
        return stats

    encoder_stats = _run_layer("encoder_out", lambda imgs: model.encode(imgs))

    proj_stats: dict = {}
    if projector_type != "none":
        # Intermediate hidden layer: up to and including the ReLU
        # mlp    → Sequential[Linear, ReLU, Linear]          → [:2]
        # mlp-bn → Sequential[Linear, BN, ReLU, Linear, BN] → [:3]
        hidden_slice = 2 if projector_type == "mlp" else 3
        proj_hidden_fn = lambda imgs, s=hidden_slice: model.projector[:s](model.encode(imgs))  # type: ignore[index]

        proj_stats = {
            **_run_layer("proj_hidden", proj_hidden_fn),
            **_run_layer("proj_out",   lambda imgs: model.projector(model.encode(imgs))),
        }

    # --- kNN accuracy ---
    knn_acc1, knn_acc5 = knn_accuracy(
        model, knn_train, val_loader, device, use_tqdm=args.tqdm, epoch=epoch,
    )

    # --- Linear probe accuracy ---
    probe_stats = eval_linear_probe(model, classifier, val_loader, device, use_tqdm=args.tqdm)

    # --- Print results ---
    metrics = {
        "epoch":           epoch,
        "knn_acc1":        knn_acc1,
        "knn_acc5":        knn_acc5,
        **probe_stats,
        **encoder_stats,
        **proj_stats,   # empty when projector == "none"
    }

    col_w = max(len(k) for k in metrics) + 2
    print(f"\n{'Metric':<{col_w}} Value")
    print("-" * (col_w + 10))
    for k, v in metrics.items():
        val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"{k:<{col_w}} {val_str}")


if __name__ == "__main__":
    main()
