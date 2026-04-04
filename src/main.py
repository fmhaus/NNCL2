"""SSL pre-training — CIFAR-100 / TinyImageNet."""

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.amp.grad_scaler import GradScaler

from dataset import load_dataset
from logger import TrainingLogger
from losses import NTXentLoss
from model import SimCLRModel, LinearClassifier

torch.set_float32_matmul_precision('high')

# ---------------------------------------------------------------------------
# Precision helpers
# ---------------------------------------------------------------------------

def make_autocast(precision: str, device: torch.device):
    if precision == "32":
        return torch.autocast(device.type, enabled=False)
    if precision in ("16", "16-mixed"):
        return torch.autocast(device.type, dtype=torch.float16)
    if precision == "bf16-mixed":
        return torch.autocast(device.type, dtype=torch.bfloat16)
    raise ValueError(f"Unknown precision '{precision}'.")


def make_scaler(precision: str, device: torch.device) -> GradScaler | None:
    if precision in ("16", "16-mixed") and device.type == "cuda":
        return GradScaler()
    return None


def make_scheduler(optimizer, total_steps: int, warmup_steps: int):
    if warmup_steps <= 0:
        return CosineAnnealingLR(optimizer, T_max=total_steps)
    warmup = LinearLR(optimizer, start_factor=1e-4, end_factor=1.0, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wrap_tqdm(iterable, use_tqdm: bool, **kwargs):
    if use_tqdm:
        from tqdm import tqdm
        return tqdm(iterable, **kwargs)
    return iterable


def _topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, topk=(1, 5)) -> list[float]:
    """Returns top-k accuracy for each k, as fractions in [0, 1]."""
    with torch.no_grad():
        maxk = max(topk)
        _, pred = logits.topk(maxk, dim=1)          # (B, maxk)
        correct = pred.t().eq(labels.view(1, -1).expand_as(pred.t()))
        return [(correct[:k].reshape(-1).float().sum() / labels.size(0)).item() for k in topk]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(model, classifier, loader, criterion, optimizer, scheduler, device, autocast, scaler, use_tqdm: bool = False, epoch: int = 0) -> dict:
    model.train()
    classifier.train()
    is_cuda = next(model.parameters()).device.type == "cuda"

    sum_nce = sum_cls = 0.0
    sum_acc1 = sum_acc5 = n_samples = 0

    for view1, view2, labels in _wrap_tqdm(loader, use_tqdm, desc=f"train [epoch {epoch}]", leave=False):
        view1  = view1.to(device, non_blocking=is_cuda)
        view2  = view2.to(device, non_blocking=is_cuda)
        labels = labels.to(device, non_blocking=is_cuda)

        with autocast:
            feat1    = model.encode(view1)
            feat2    = model.encode(view2)
            nce_loss = criterion(model.projector(feat1), model.projector(feat2))
            # Classifier on both views averaged — 2× gradient signal per step
            logits1  = classifier(feat1.detach())
            logits2  = classifier(feat2.detach())
            cls_loss = (F.cross_entropy(logits1, labels) + F.cross_entropy(logits2, labels)) / 2
            loss     = nce_loss + cls_loss

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        scheduler.step()

        acc1, acc5 = _topk_accuracy(logits1.detach(), labels)
        b          = labels.size(0)
        sum_nce   += nce_loss.item()
        sum_cls   += cls_loss.item()
        sum_acc1  += acc1 * b
        sum_acc5  += acc5 * b
        n_samples += b

    n = len(loader)
    return {
        "train_nce_loss":   sum_nce  / n,
        "train_class_loss": sum_cls  / n,
        "train_acc1_epoch": sum_acc1 / n_samples,
        "train_acc5_epoch": sum_acc5 / n_samples,
    }


# ---------------------------------------------------------------------------
# Evaluation — linear classifier on val set
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_classifier(model, classifier, loader, device, use_tqdm: bool = False, epoch: int = 0) -> dict:
    model.eval()
    classifier.eval()
    is_cuda = next(model.parameters()).device.type == "cuda"

    sum_loss = sum_acc1 = sum_acc5 = n_samples = 0

    for images, labels in _wrap_tqdm(loader, use_tqdm, desc=f"val [epoch {epoch}]", leave=False):
        images = images.to(device, non_blocking=is_cuda)
        labels = labels.to(device, non_blocking=is_cuda)

        logits    = classifier(model.encode(images))
        sum_loss += F.cross_entropy(logits, labels).item()

        acc1, acc5 = _topk_accuracy(logits, labels)
        b          = labels.size(0)
        sum_acc1  += acc1 * b
        sum_acc5  += acc5 * b
        n_samples += b

    return {
        "val_loss": sum_loss / len(loader),
        "val_acc1": sum_acc1 / n_samples,
        "val_acc5": sum_acc5 / n_samples,
    }


# ---------------------------------------------------------------------------
# Evaluation — kNN top-1 and top-5
# ---------------------------------------------------------------------------

@torch.no_grad()
def knn_accuracy(
    model, train_loader, val_loader, device,
    k: int = 20, temperature: float = 0.07, use_tqdm: bool = False, epoch: int = 0,
) -> tuple[float, float]:
    """Returns (top-1 kNN acc, top-5 kNN acc)."""
    model.eval()
    is_cuda = next(model.parameters()).device.type == "cuda"

    def extract(loader, desc):
        feats, labels = [], []
        for images, targets in _wrap_tqdm(loader, use_tqdm, desc=desc, leave=False):
            feats.append(model.encode(images.to(device, non_blocking=is_cuda)))
            labels.append(targets.to(device, non_blocking=is_cuda))
        return torch.cat(feats), torch.cat(labels)

    train_feats, train_labels = extract(train_loader, f"knn-train [epoch {epoch}]")
    val_feats,   val_labels   = extract(val_loader,   f"knn-val   [epoch {epoch}]")

    train_feats = nn.functional.normalize(train_feats, dim=1)
    val_feats   = nn.functional.normalize(val_feats,   dim=1)

    num_classes = int(train_labels.max().item()) + 1
    correct1 = correct5 = 0

    for vf, vl in zip(val_feats.split(512), val_labels.split(512)):
        top_k      = (vf @ train_feats.T / temperature).topk(k, dim=1).indices  # (chunk, k)
        knn_labels = train_labels[top_k]                                         # (chunk, k)

        # Aggregate class votes, then rank
        votes = torch.zeros(vl.size(0), num_classes, device=device)
        votes.scatter_add_(1, knn_labels, torch.ones_like(knn_labels, dtype=torch.float))
        _, top5 = votes.topk(5, dim=1)                                           # (chunk, 5)

        correct1 += (top5[:, 0] == vl).sum().item()
        correct5 += (top5 == vl.unsqueeze(1)).any(dim=1).sum().item()

    total = len(val_labels)
    return correct1 / total, correct5 / total


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _checkpoint_state(epoch, model, classifier, optimizer, scheduler, scaler, args) -> dict:
    state = {
        "epoch":      epoch,
        "model":      model.state_dict(),
        "classifier": classifier.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "scheduler":  scheduler.state_dict(),
        "args":       vars(args),
    }
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    return state


def load_checkpoint(path, model, classifier, optimizer, scheduler, scaler) -> int:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    if "classifier" in ckpt:
        classifier.load_state_dict(ckpt["classifier"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    print(f"Resumed from epoch {ckpt['epoch']}")
    return ckpt["epoch"]


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="SSL pre-training (SimCLR) on CIFAR-100 or TinyImageNet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--name",            required=True,
                   help="Run name. Results saved to saves/<name>/.")
    p.add_argument("--method",          default="simclr",  choices=["simclr", "byol"],
                   help="SSL method. byol is not yet implemented.")
    p.add_argument("--dataset",         default="cifar100", choices=["cifar100", "tinyimagenet"])
    p.add_argument("--data-root",       default="./data")
    p.add_argument("--num-workers",     default=4,          type=int)
    p.add_argument("--proj-hidden-dim",    default=2048,        type=int)
    p.add_argument("--proj-output-dim",    default=128,        type=int)
    p.add_argument("--no-projector",       action="store_true",
                   help="Disable the projection head (apply loss directly on backbone features).")
    p.add_argument("--feature-transform",  default=None,
                   choices=["relu", "softmax", "L1_norm"],
                   help="Non-negative transform applied to backbone features before projector and downstream tasks.")
    p.add_argument("--pred-hidden-dim", default=512,        type=int,
                   help="BYOL predictor hidden dim (reserved).")
    p.add_argument("--max-epochs",      default=200,        type=int)
    p.add_argument("--warmup-epochs",   default=10,         type=int)
    p.add_argument("--batch-size",      default=256,        type=int)
    p.add_argument("--lr",              default=None,       type=float,
                   help="Learning rate. Default: 0.3 * batch_size / 256.")
    p.add_argument("--weight-decay",     default=1e-4,       type=float)

    p.add_argument("--classifier-lr",   default=0.1,        type=float,
                   help="Learning rate for the online linear classifier optimizer.")
    p.add_argument("--temperature",     default=0.1,        type=float,
                   help="NT-Xent temperature. Paper: 0.1 for CIFAR, 0.07 for ImageNet.")
    p.add_argument("--precision",       default="32",       choices=["32", "16", "16-mixed", "bf16-mixed"])
    p.add_argument("--seed",            default=42,         type=int)
    p.add_argument("--resume",          action="store_true",
                   help="Resume training from saves/<name>/. Overrides all args from hparams.json.")
    p.add_argument("--compile",          action="store_true",
                   help="torch.compile the model and classifier.")
    p.add_argument("--console-log",     action="store_true",
                   help="Print metrics to console after every epoch.")
    p.add_argument("--tqdm",            action="store_true",
                   help="Show tqdm progress bars inside each epoch.")

    args = p.parse_args()
    if args.method == "byol":
        p.error("--method byol is not yet implemented.")
    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args     = parse_args()
    save_dir = Path("saves") / args.name

    # --- Existence checks ---
    if args.resume:
        if not save_dir.exists():
            raise SystemExit(f"Cannot resume: '{save_dir}' does not exist.")
        ckpts = sorted(save_dir.glob("state_*.ckpt"))
        if not ckpts:
            raise SystemExit(f"Cannot resume: no checkpoints found in '{save_dir}'.")

        # Override all args from the saved run before any setup
        hparams = json.loads((save_dir / "hparams.json").read_text())
        saved_name = args.name
        for key, value in hparams.items():
            setattr(args, key, value)
        args.name   = saved_name  # keep current invocation's name
        args.resume = True        # hparams likely has resume=False

        resume_ckpt = ckpts[-1]
        print(f"Resuming '{args.name}' from {resume_ckpt.name}")
    else:
        if save_dir.exists():
            raise SystemExit(f"Run '{args.name}' already exists at '{save_dir}'. Use --resume to continue it.")
        resume_ckpt = None

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_cuda = device.type == "cuda"

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if is_cuda:
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True  # auto-tune kernels for fixed input sizes

    if args.lr is None:
        args.lr = 0.3 * args.batch_size / 256

    logger = TrainingLogger(save_dir, args, console_log=args.console_log)

    print(f"Method: {args.method} | Dataset: {args.dataset} | Device: {device} | Precision: {args.precision}")

    # --- Data ---
    if args.dataset == "cifar100":
        image_size, num_classes = 32, 100
    else:
        image_size, num_classes = 64, 200

    def get_data_loader(two_view: bool, augment, train: bool):
        return load_dataset(
            args.dataset, two_view=two_view, augment=augment, train=train,
            batch_size=args.batch_size, num_workers=args.num_workers, data_root=args.data_root,
        )

    train_loader = get_data_loader(two_view=True,  augment=args.method, train=True)
    knn_train    = get_data_loader(two_view=False, augment=None,        train=True)
    knn_val      = get_data_loader(two_view=False, augment=None,        train=False)

    # --- Model, classifier, loss, optimiser ---
    model      = SimCLRModel(proj_hidden=args.proj_hidden_dim, proj_dim=args.proj_output_dim, image_size=image_size, use_projector=not args.no_projector, feature_transform=args.feature_transform).to(device)
    classifier = LinearClassifier(num_classes=num_classes).to(device)
    criterion  = NTXentLoss(temperature=args.temperature)

    # Exclude normalization from weight decay
    _norm_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)
    decay, no_decay = [], []
    for module in model.modules():
        if isinstance(module, _norm_types):
            no_decay.extend(module.parameters(recurse=False))
        else:
            decay.extend(p for p in module.parameters(recurse=False) if p.requires_grad)

    optimizer = SGD(
        [
            {"params": decay},
            {"params": no_decay,                  "weight_decay": 0.0},
            {"params": classifier.parameters(),   "weight_decay": 0.0, "lr": args.classifier_lr},
        ],
        lr=args.lr, momentum=0.9, weight_decay=args.weight_decay,
    )
    steps_per_epoch = len(train_loader)
    scheduler = make_scheduler(optimizer, args.max_epochs * steps_per_epoch, args.warmup_epochs * steps_per_epoch)
    autocast  = make_autocast(args.precision, device)
    scaler    = make_scaler(args.precision, device)

    start_epoch = 0
    if resume_ckpt is not None:
        start_epoch = load_checkpoint(resume_ckpt, model, classifier, optimizer, scheduler, scaler)

    # Keep uncompiled references for state dict I/O — torch.compile wraps the
    # module and _model.state_dict() returns mangled keys.
    model_for_ckpt      = model
    classifier_for_ckpt = classifier
    if args.compile:
        model      = torch.compile(model)
        classifier = torch.compile(classifier)

    # --- Training loop ---
    for epoch in range(start_epoch, args.max_epochs):
        t0           = time.perf_counter()
        train_m    = train_epoch(model, classifier, train_loader, criterion, optimizer, scheduler, device, autocast, scaler, args.tqdm, epoch=epoch + 1)
        knn_acc1, knn_acc5 = knn_accuracy(model, knn_train, knn_val, device, use_tqdm=args.tqdm, epoch=epoch + 1)
        val_m              = eval_classifier(model, classifier, knn_val, device, use_tqdm=args.tqdm, epoch=epoch + 1)
        epoch_time   = time.perf_counter() - t0

        metrics = {
            "train_nce_loss":   train_m["train_nce_loss"],
            "train_class_loss": train_m["train_class_loss"],
            "train_acc1_epoch": train_m["train_acc1_epoch"],
            "train_acc5_epoch": train_m["train_acc5_epoch"],
            "val_knn_acc1":     knn_acc1,
            "val_knn_acc5":     knn_acc5,
            "val_loss":         val_m["val_loss"],
            "val_acc1":         val_m["val_acc1"],
            "val_acc5":         val_m["val_acc5"],
            "lr":               scheduler.get_last_lr()[0],
            "epoch_time_s":     epoch_time,
        }
        logger.log(epoch + 1, metrics)
        logger.save_checkpoint(epoch + 1, _checkpoint_state(epoch + 1, model_for_ckpt, classifier_for_ckpt, optimizer, scheduler, scaler, args))

    print("Training complete.")


if __name__ == "__main__":
    main()
