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

    # --- Layer evaluations ---
    encoder_stats = eval_layer(
        layer_fn=lambda imgs: model.encode(imgs),
        loader=val_loader,
        name="encoder_out",
        logger=logger,
        epoch=epoch,
        device=device,
    )

    proj_stats = eval_layer(
        layer_fn=lambda imgs: model.projector(model.encode(imgs)),
        loader=val_loader,
        name="proj_out",
        logger=logger,
        epoch=epoch,
        device=device,
    )

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
        **proj_stats,
    }

    col_w = max(len(k) for k in metrics) + 2
    print(f"\n{'Metric':<{col_w}} Value")
    print("-" * (col_w + 10))
    for k, v in metrics.items():
        val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"{k:<{col_w}} {val_str}")


if __name__ == "__main__":
    main()
