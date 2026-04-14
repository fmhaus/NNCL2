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
from evaluator import evaluate_features, evaluate_features_fast
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

def train_epoch(model, classifiers, eval_names, loader, criterion, optimizer, scheduler, device, autocast, scaler, use_tqdm: bool = False, epoch: int = 0) -> dict:
    model.train()
    for clf in classifiers:
        clf.train()
    is_cuda = next(model.parameters()).device.type == "cuda"

    n_eval    = len(eval_names)
    sum_nce   = sum_cls = 0.0
    sum_acc1  = [0.0] * n_eval
    sum_grad  = [0.0] * n_eval
    n_samples = 0

    for view1, view2, labels in _wrap_tqdm(loader, use_tqdm, desc=f"train [epoch {epoch}]", leave=False):
        view1  = view1.to(device, non_blocking=is_cuda)
        view2  = view2.to(device, non_blocking=is_cuda)
        labels = labels.to(device, non_blocking=is_cuda)

        _grads: list[list[float]] = [[] for _ in range(n_eval)]

        with autocast:
            feats1     = model.encode_all(view1)
            feats2     = model.encode_all(view2)
            nce_loss   = criterion(feats1[-1], feats2[-1])
            eval_f1    = [feats1[0], feats1[-1]]
            eval_f2    = [feats2[0], feats2[-1]]
            # Classifier on both views averaged — 2× gradient signal per step
            logits1    = [clf(f.detach()) for clf, f in zip(classifiers, eval_f1)]
            logits2    = [clf(f.detach()) for clf, f in zip(classifiers, eval_f2)]
            cls_loss   = torch.stack([
                (F.cross_entropy(l1, labels) + F.cross_entropy(l2, labels)) / 2
                for l1, l2 in zip(logits1, logits2)
            ]).mean()
            loss = nce_loss + cls_loss

        hooks = [
            f.register_hook(lambda g, i=i: _grads[i].append(g.norm().item()))
            for i, f in enumerate(eval_f1)
        ]

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        scheduler.step()

        for hook in hooks:
            hook.remove()

        b = labels.size(0)
        sum_nce += nce_loss.item()
        sum_cls += cls_loss.item()
        for i, logit in enumerate(logits1):
            sum_acc1[i] += _topk_accuracy(logit.detach(), labels, topk=(1,))[0] * b
        for i in range(n_eval):
            sum_grad[i] += _grads[i][0] if _grads[i] else 0.0
        n_samples += b

    n = len(loader)
    result: dict = {
        "train_nce_loss": sum_nce / n,
        "train_cls_loss": sum_cls / n,
    }
    for i, name in enumerate(eval_names):
        result[f"{name}_grad"]       = sum_grad[i] / n
        result[f"{name}_train_acc1"] = sum_acc1[i] / n_samples
    return result


# ---------------------------------------------------------------------------
# Evaluation — linear classifier on val set
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_eval_features(model, loader, device, use_tqdm: bool = False, desc: str = "") -> tuple[list[list], list]:
    """Returns ([backbone_feats, head_feats], labels)."""
    model.eval()
    is_cuda = next(model.parameters()).device.type == "cuda"
    backbone_feats: list = []
    head_feats:     list = []
    labels = []
    for images, targets in _wrap_tqdm(loader, use_tqdm, desc=desc, leave=False):
        enc = model.encode_all(images.to(device, non_blocking=is_cuda))
        backbone_feats.append(enc[0])
        head_feats.append(enc[-1])
        labels.append(targets.to(device, non_blocking=is_cuda))
    return [backbone_feats, head_feats], labels


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _checkpoint_state(epoch, model, classifiers, optimizer, scheduler, scaler, args) -> dict:
    state = {
        "epoch":       epoch,
        "model":       model.state_dict(),
        "classifiers": classifiers.state_dict(),
        "optimizer":   optimizer.state_dict(),
        "scheduler":   scheduler.state_dict(),
        "args":        vars(args),
    }
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    return state


def load_checkpoint(path, model, classifiers, optimizer, scheduler, scaler) -> int:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    if "classifiers" in ckpt:
        classifiers.load_state_dict(ckpt["classifiers"])
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
    p.add_argument("--dataset",         default="cifar100", choices=["cifar100", "tinyimagenet"])
    p.add_argument("--data-root",       default="./data")
    p.add_argument("--num-workers",     default=4,          type=int)
    p.add_argument("--proj-out-dim",  default=128, type=int,
                   help="Output dimension of the projection head.")
    p.add_argument("--proj-layers",   default=1,   type=int,
                   help="Hidden x→x blocks before the final compression. 0 = head only.")
    p.add_argument("--no-projector",  action="store_true",
                   help="Disable the projection head entirely.")
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
    def _cosine_sim(v):
        v = float(v)
        if not (-1.0 <= v <= 1.0):
            raise argparse.ArgumentTypeError(f"--sim-clip-min must be in [-1, 1], got {v}")
        return v
    p.add_argument("--sim-clip-min",    default=-1.0,       type=_cosine_sim,
                   help="Cosine similarity clip threshold for NT-Xent negatives. -1 = no clipping, 0 = clip at 90°.")
    p.add_argument("--precision",       default="32",       choices=["32", "16", "16-mixed", "bf16-mixed"])
    p.add_argument("--seed",            default=42,         type=int)
    p.add_argument("--resume",          action="store_true",
                   help="Resume training from saves/<name>/. Overrides all args from hparams.json.")
    p.add_argument("--compile",          action="store_true",
                   help="torch.compile the model and classifier.")
    p.add_argument("--eval-freq",        default=10,         type=int,
                   help="Run full evaluation (MIG, ortho, sparsity) every N epochs. KNN and classifier run every epoch.")
    p.add_argument("--console-log",     action="store_true",
                   help="Print metrics to console after every epoch.")
    p.add_argument("--tqdm",            action="store_true",
                   help="Show tqdm progress bars inside each epoch.")

    return p.parse_args()


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
            raise SystemExit(f"Run '{args.name}' already exists at '{save_dir}'. Use --resume to continue it                                            .")
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

    print(f"Dataset: {args.dataset} | Device: {device} | Precision: {args.precision}")

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

    train_loader = get_data_loader(two_view=True,  augment="simclr", train=True)
    knn_train    = get_data_loader(two_view=False, augment=None,        train=True)
    knn_val      = get_data_loader(two_view=False, augment=None,        train=False)

    # --- Model, classifiers, loss, optimiser ---
    proj_layers = None if args.no_projector else args.proj_layers
    model       = SimCLRModel(proj_out_dim=args.proj_out_dim, proj_layers=proj_layers, image_size=image_size).to(device)
    all_names   = model.feature_names
    all_dims    = model.feature_dims
    eval_names  = [all_names[0], all_names[-1]]   # backbone, head
    eval_dims   = [all_dims[0],  all_dims[-1]]
    classifiers = nn.ModuleList(
        LinearClassifier(dim, num_classes) for dim in eval_dims
    ).to(device)
    criterion  = NTXentLoss(temperature=args.temperature, sim_clip_min=args.sim_clip_min)

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
            {"params": no_decay,                   "weight_decay": 0.0},
            {"params": classifiers.parameters(),   "weight_decay": 0.0, "lr": args.classifier_lr},
        ],
        lr=args.lr, momentum=0.9, weight_decay=args.weight_decay,
    )
    steps_per_epoch = len(train_loader)
    scheduler = make_scheduler(optimizer, args.max_epochs * steps_per_epoch, args.warmup_epochs * steps_per_epoch)
    autocast  = make_autocast(args.precision, device)
    scaler    = make_scaler(args.precision, device)

    start_epoch = 0
    if resume_ckpt is not None:
        start_epoch = load_checkpoint(resume_ckpt, model, classifiers, optimizer, scheduler, scaler)

    # Keep uncompiled references for state dict I/O — torch.compile wraps the
    # module and _model.state_dict() returns mangled keys.
    model_for_ckpt       = model
    classifiers_for_ckpt = classifiers
    if args.compile:
        model = torch.compile(model)

    # --- Training loop ---
    for epoch in range(start_epoch, args.max_epochs):
        t0      = time.perf_counter()
        train_m = train_epoch(model, classifiers, eval_names, train_loader, criterion, optimizer, scheduler, device, autocast, scaler, args.tqdm, epoch=epoch + 1)

        train_feats, train_labels = extract_eval_features(model, knn_train, device, args.tqdm, desc=f"knn-train [epoch {epoch + 1}]")
        val_feats,   val_labels   = extract_eval_features(model, knn_val,   device, args.tqdm, desc=f"knn-val   [epoch {epoch + 1}]")

        full_eval = (epoch + 1) % args.eval_freq == 0 or (epoch + 1) == args.max_epochs
        eval_fn   = evaluate_features if full_eval else evaluate_features_fast

        val_m = {}
        for i, name in enumerate(eval_names):
            m = eval_fn(train_feats[i], train_labels, val_feats[i], val_labels, classifiers_for_ckpt[i])
            val_m.update({f"{name}_{k}": v for k, v in m.items()})

        epoch_time = time.perf_counter() - t0

        metrics = {
            "train_nce_loss": train_m["train_nce_loss"],
            "train_cls_loss": train_m["train_cls_loss"],
            "lr":             scheduler.get_last_lr()[0],
            "epoch_time_s":   epoch_time,
            **{k: v for k, v in train_m.items() if k not in ("train_nce_loss", "train_cls_loss")},
            **val_m,
        }
        logger.log(epoch + 1, metrics)
        logger.save_checkpoint(epoch + 1, _checkpoint_state(epoch + 1, model_for_ckpt, classifiers_for_ckpt, optimizer, scheduler, scaler, args))

    print("Training complete.")


if __name__ == "__main__":
    main()
