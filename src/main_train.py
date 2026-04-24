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

def train_epoch(
    model, backbone_classifier, proj_classifier,
    loader, criterion, optimizer, scheduler,
    device, autocast, scaler,
    has_projector: bool,
    use_tqdm: bool = False, epoch: int = 0,
) -> dict:
    model.train()
    backbone_classifier.train()
    if proj_classifier is not None:
        proj_classifier.train()
    is_cuda = next(model.parameters()).device.type == "cuda"

    sum_nce = sum_backbone_cls = sum_proj_cls = 0.0
    sum_acc1 = sum_acc5 = sum_proj_acc1 = sum_proj_acc5 = n_samples = 0
    sum_grad_feat = sum_grad_proj_out = 0.0

    for view1, view2, labels in _wrap_tqdm(loader, use_tqdm, desc=f"train [epoch {epoch}]", leave=False):
        view1  = view1.to(device, non_blocking=is_cuda)
        view2  = view2.to(device, non_blocking=is_cuda)
        labels = labels.to(device, non_blocking=is_cuda)

        _grad_feat     = []
        _grad_proj_out = []

        with autocast:
            feat1     = model.encode(view1)
            feat2     = model.encode(view2)
            proj_out1 = model.projector(feat1)
            proj_out2 = model.projector(feat2)
            nce_loss  = criterion(proj_out1, proj_out2)
            # Backbone classifier on both views averaged — 2× gradient signal per step
            logits1          = backbone_classifier(feat1.detach())
            logits2          = backbone_classifier(feat2.detach())
            backbone_cls_loss = (F.cross_entropy(logits1, labels) + F.cross_entropy(logits2, labels)) / 2

            if has_projector:
                proj_logits1  = proj_classifier(proj_out1.detach())
                proj_logits2  = proj_classifier(proj_out2.detach())
                proj_cls_loss = (F.cross_entropy(proj_logits1, labels) + F.cross_entropy(proj_logits2, labels)) / 2
                loss          = nce_loss + backbone_cls_loss + proj_cls_loss
            else:
                proj_logits1  = None
                proj_cls_loss = nce_loss.new_tensor(0.0)
                loss          = nce_loss + backbone_cls_loss

        feat_hook     = feat1.register_hook(lambda g: _grad_feat.append(g.norm().item()))
        proj_out_hook = proj_out1.register_hook(lambda g: _grad_proj_out.append(g.norm().item()))

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        scheduler.step()

        feat_hook.remove()
        proj_out_hook.remove()

        sum_grad_feat     += _grad_feat[0]     if _grad_feat     else 0.0
        sum_grad_proj_out += _grad_proj_out[0] if _grad_proj_out else 0.0

        acc1, acc5 = _topk_accuracy(logits1.detach(), labels)
        b = labels.size(0)
        sum_nce          += nce_loss.item()
        sum_backbone_cls += backbone_cls_loss.item()
        sum_acc1         += acc1 * b
        sum_acc5         += acc5 * b
        if has_projector and proj_logits1 is not None:
            proj_acc1, proj_acc5 = _topk_accuracy(proj_logits1.detach(), labels)
            sum_proj_cls  += proj_cls_loss.item()
            sum_proj_acc1 += proj_acc1 * b
            sum_proj_acc5 += proj_acc5 * b
        n_samples += b

    n = len(loader)
    result = {
        "train_nce_loss":          sum_nce          / n,
        "train_backbone_cls_loss": sum_backbone_cls / n,
        "grad_feat_norm":          sum_grad_feat     / n,
        "grad_proj_out_norm":      sum_grad_proj_out / n,
        "train_backbone_acc1":     sum_acc1 / n_samples,
        "train_backbone_acc5":     sum_acc5 / n_samples,
    }
    if has_projector:
        result["train_proj_cls_loss"] = sum_proj_cls  / n
        result["train_proj_acc1"]     = sum_proj_acc1 / n_samples
        result["train_proj_acc5"]     = sum_proj_acc5 / n_samples
    return result


# ---------------------------------------------------------------------------
# Evaluation — linear classifier on val set
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_classifier(
    model, feature_fn, classifier, loader, device,
    prefix: str = "backbone", use_tqdm: bool = False, epoch: int = 0,
) -> dict:
    model.eval()
    classifier.eval()
    is_cuda = next(model.parameters()).device.type == "cuda"

    sum_loss = sum_acc1 = sum_acc5 = sum_l1 = sum_l2 = sum_hoyer = n_samples = 0

    for images, labels in _wrap_tqdm(loader, use_tqdm, desc=f"val [{prefix}] [epoch {epoch}]", leave=False):
        images = images.to(device, non_blocking=is_cuda)
        labels = labels.to(device, non_blocking=is_cuda)

        feats     = feature_fn(images)
        logits    = classifier(feats)
        sum_loss += F.cross_entropy(logits, labels).item()

        acc1, acc5 = _topk_accuracy(logits, labels)
        b          = labels.size(0)
        d          = feats.size(1)
        l1         = feats.norm(p=1, dim=1)   # (B,)
        l2         = feats.norm(p=2, dim=1)   # (B,)
        sum_acc1  += acc1 * b
        sum_acc5  += acc5 * b
        sum_l1    += l1.mean().item() * b
        sum_l2    += l2.mean().item() * b
        sum_hoyer += ((d ** 0.5 - l1 / l2.clamp(min=1e-8)) / (d ** 0.5 - 1)).mean().item() * b
        n_samples += b

    return {
        f"{prefix}_val_loss":   sum_loss  / len(loader),
        f"{prefix}_val_acc1":   sum_acc1  / n_samples,
        f"{prefix}_val_acc5":   sum_acc5  / n_samples,
        f"{prefix}_feat_l1":    sum_l1   / n_samples,
        f"{prefix}_feat_l2":    sum_l2   / n_samples,
        f"{prefix}_feat_hoyer": sum_hoyer / n_samples,
    }


# ---------------------------------------------------------------------------
# Evaluation — kNN top-1 and top-5
# ---------------------------------------------------------------------------

@torch.no_grad()
def knn_accuracy(
    model, feature_fn, train_loader, val_loader, device,
    k: int = 20, temperature: float = 0.07, use_tqdm: bool = False, epoch: int = 0,
) -> tuple[float, float]:
    """Returns (top-1 kNN acc, top-5 kNN acc)."""
    model.eval()
    is_cuda = next(model.parameters()).device.type == "cuda"

    def extract(loader, desc):
        feats, labels = [], []
        for images, targets in _wrap_tqdm(loader, use_tqdm, desc=desc, leave=False):
            feats.append(feature_fn(images.to(device, non_blocking=is_cuda)))
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

def _checkpoint_state(epoch, model, backbone_classifier, proj_classifier, optimizer, scheduler, scaler, args) -> dict:
    state = {
        "epoch":               epoch,
        "model":               model.state_dict(),
        "backbone_classifier": backbone_classifier.state_dict(),
        "optimizer":           optimizer.state_dict(),
        "scheduler":           scheduler.state_dict(),
        "args":                vars(args),
    }
    if proj_classifier is not None:
        state["proj_classifier"] = proj_classifier.state_dict()
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    return state


def load_checkpoint(path, model, backbone_classifier, proj_classifier, optimizer, scheduler, scaler) -> int:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    if "backbone_classifier" in ckpt:
        backbone_classifier.load_state_dict(ckpt["backbone_classifier"])
    if proj_classifier is not None and "proj_classifier" in ckpt:
        proj_classifier.load_state_dict(ckpt["proj_classifier"])
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
    p.add_argument("--method",          default="simclr",  choices=["simclr"],
                   help="SSL method.")
    p.add_argument("--dataset",         default="cifar100", choices=["cifar100", "tinyimagenet"])
    p.add_argument("--data-root",       default="./data")
    p.add_argument("--num-workers",     default=4,          type=int)
    p.add_argument("--proj-hidden-dim", default=512,        type=int)
    p.add_argument("--proj-output-dim", default=128,        type=int)
    p.add_argument("--projector",       default="mlp",      choices=["none", "mlp", "mlp-bn"],
                   help="Projection head variant: none (identity), mlp (linear-relu-linear), mlp-bn (SimCLR paper, BN after each linear).")
    p.add_argument("--non-neg",         action="store_true", dest="non_neg",
                   help="Append ReLUGeLUGrad to projector end (non-negative features). No-op when --projector none.")
    p.add_argument("--max-epochs",      default=200,        type=int)
    p.add_argument("--warmup-epochs",   default=10,         type=int)
    p.add_argument("--batch-size",      default=256,        type=int)
    p.add_argument("--lr",              default=None,       type=float,
                   help="Learning rate. Default: 0.3 * batch_size / 256.")
    p.add_argument("--weight-decay",    default=1e-4,       type=float)
    p.add_argument("--classifier-lr",  default=0.1,        type=float,
                   help="Learning rate for the online linear classifier optimizer.")
    p.add_argument("--temperature",     default=0.1,        type=float,
                   help="NT-Xent temperature. SimCLR paper optimum: 0.5.")
    p.add_argument("--precision",       default="32",       choices=["32", "16", "16-mixed", "bf16-mixed"])
    p.add_argument("--seed",            default=42,         type=int)
    p.add_argument("--resume",          action="store_true",
                   help="Resume training from saves/<name>/. Overrides all args from hparams.json.")
    p.add_argument("--no-compile",      action="store_false", dest="compile",
                   help="Disable torch.compile.")
    p.add_argument("--no-console-log",  action="store_false", dest="console_log",
                   help="Suppress per-epoch metric printing.")
    p.add_argument("--no-tqdm",         action="store_false", dest="tqdm",
                   help="Disable tqdm progress bars inside each epoch.")
    p.set_defaults(compile=True, console_log=True, tqdm=True, non_neg=False)

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

    # --- Model, classifiers, loss, optimiser ---
    has_projector = args.projector != "none"
    non_neg       = getattr(args, "non_neg", False)

    model               = SimCLRModel(
        proj_hidden=args.proj_hidden_dim, proj_dim=args.proj_output_dim,
        image_size=image_size, projector=args.projector, non_neg=non_neg,
    ).to(device)
    backbone_classifier = LinearClassifier(num_classes=num_classes).to(device)
    proj_classifier     = (
        LinearClassifier(num_classes=num_classes, input_dim=args.proj_output_dim).to(device)
        if has_projector else None
    )
    criterion = NTXentLoss(temperature=args.temperature)

    # Exclude normalization layers and biases from weight decay
    _norm_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)
    decay, no_decay = [], []
    for module in model.modules():
        if isinstance(module, _norm_types):
            no_decay.extend(module.parameters(recurse=False))
        else:
            for name, p in module.named_parameters(recurse=False):
                if not p.requires_grad:
                    continue
                if name == "bias":
                    no_decay.append(p)
                else:
                    decay.append(p)

    param_groups = [
        {"params": decay},
        {"params": no_decay,                         "weight_decay": 0.0},
        {"params": backbone_classifier.parameters(), "weight_decay": 0.0, "lr": args.classifier_lr},
    ]
    if has_projector:
        param_groups.append({"params": proj_classifier.parameters(), "weight_decay": 0.0, "lr": args.classifier_lr})

    optimizer = SGD(param_groups, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    steps_per_epoch = len(train_loader)
    scheduler = make_scheduler(optimizer, args.max_epochs * steps_per_epoch, args.warmup_epochs * steps_per_epoch)
    autocast  = make_autocast(args.precision, device)
    scaler    = make_scaler(args.precision, device)

    start_epoch = 0
    if resume_ckpt is not None:
        start_epoch = load_checkpoint(resume_ckpt, model, backbone_classifier, proj_classifier, optimizer, scheduler, scaler)

    # Keep uncompiled references for state dict I/O — torch.compile wraps the
    # module and _model.state_dict() returns mangled keys.
    model_for_ckpt               = model
    backbone_classifier_for_ckpt = backbone_classifier
    proj_classifier_for_ckpt     = proj_classifier
    if args.compile:
        model               = torch.compile(model)
        backbone_classifier = torch.compile(backbone_classifier)
        if proj_classifier is not None:
            proj_classifier = torch.compile(proj_classifier)

    # --- Training loop ---
    for epoch in range(start_epoch, args.max_epochs):
        t0      = time.perf_counter()
        train_m = train_epoch(
            model, backbone_classifier, proj_classifier, train_loader, criterion,
            optimizer, scheduler, device, autocast, scaler,
            has_projector=has_projector, use_tqdm=args.tqdm, epoch=epoch + 1,
        )
        backbone_knn_acc1, backbone_knn_acc5 = knn_accuracy(
            model, model.encode, knn_train, knn_val, device, use_tqdm=args.tqdm, epoch=epoch + 1,
        )
        backbone_val_m = eval_classifier(
            model, model.encode, backbone_classifier, knn_val, device,
            prefix="backbone", use_tqdm=args.tqdm, epoch=epoch + 1,
        )
        if has_projector:
            proj_knn_acc1, proj_knn_acc5 = knn_accuracy(
                model, model, knn_train, knn_val, device, use_tqdm=args.tqdm, epoch=epoch + 1,
            )
            proj_val_m = eval_classifier(
                model, model, proj_classifier, knn_val, device,
                prefix="proj", use_tqdm=args.tqdm, epoch=epoch + 1,
            )
        epoch_time = time.perf_counter() - t0

        metrics = {
            "train_nce_loss":          train_m["train_nce_loss"],
            "train_backbone_cls_loss": train_m["train_backbone_cls_loss"],
            "train_backbone_acc1":     train_m["train_backbone_acc1"],
            "train_backbone_acc5":     train_m["train_backbone_acc5"],
            "backbone_knn_acc1":       backbone_knn_acc1,
            "backbone_knn_acc5":       backbone_knn_acc5,
            "grad_feat_norm":          train_m["grad_feat_norm"],
            "grad_proj_out_norm":      train_m["grad_proj_out_norm"],
            **backbone_val_m,
            "lr":                      scheduler.get_last_lr()[0],
            "epoch_time_s":            epoch_time,
        }
        if has_projector:
            metrics.update({
                "train_proj_cls_loss": train_m["train_proj_cls_loss"],
                "train_proj_acc1":     train_m["train_proj_acc1"],
                "train_proj_acc5":     train_m["train_proj_acc5"],
                "proj_knn_acc1":       proj_knn_acc1,
                "proj_knn_acc5":       proj_knn_acc5,
                **proj_val_m,
            })
        logger.log(epoch + 1, metrics)
        logger.save_checkpoint(
            epoch + 1,
            _checkpoint_state(epoch + 1, model_for_ckpt, backbone_classifier_for_ckpt, proj_classifier_for_ckpt, optimizer, scheduler, scaler, args),
        )

    print("Training complete.")


if __name__ == "__main__":
    main()
