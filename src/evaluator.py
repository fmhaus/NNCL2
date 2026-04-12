"""Evaluation utilities that operate on lists of pre-extracted feature tensors."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def knn_eval(
    train_feats:  list[torch.Tensor],
    train_labels: list[torch.Tensor],
    val_feats:    list[torch.Tensor],
    val_labels:   list[torch.Tensor],
    k: int = 20,
    temperature: float = 0.07,
    chunk_size: int = 512,
) -> dict[str, float]:
    """kNN classifier evaluation on pre-extracted features.

    Returns:
        {"knn_acc1": float, "knn_acc5": float,
         "train_feat_l1": float, "train_feat_l2": float,
         "val_feat_l1":   float, "val_feat_l2":   float}
    """
    tf_raw = torch.cat(train_feats).float()
    vf_raw = torch.cat(val_feats).float()
    tf = F.normalize(tf_raw, dim=1)
    vf = F.normalize(vf_raw, dim=1)
    tl = torch.cat(train_labels)
    vl = torch.cat(val_labels)

    device = tf.device
    num_classes = int(tl.max().item()) + 1
    correct1 = correct5 = 0

    for vf_chunk, vl_chunk in zip(vf.split(chunk_size), vl.split(chunk_size)):
        top_k      = (vf_chunk @ tf.T / temperature).topk(k, dim=1).indices  # (chunk, k)
        knn_labels = tl[top_k]                                                 # (chunk, k)

        votes = torch.zeros(vl_chunk.size(0), num_classes, device=device)
        votes.scatter_add_(1, knn_labels, torch.ones_like(knn_labels, dtype=torch.float))
        _, top5 = votes.topk(min(5, num_classes), dim=1)                       # (chunk, 5)

        correct1 += (top5[:, 0] == vl_chunk).sum().item()
        correct5 += (top5 == vl_chunk.unsqueeze(1)).any(dim=1).sum().item()

    total = vl.size(0)
    return {
        "knn_acc1": correct1 / total,
        "knn_acc5": correct5 / total,
    }


def _topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, topk=(1, 5)) -> list[float]:
    with torch.no_grad():
        maxk = max(topk)
        _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
        correct = pred.eq(labels.unsqueeze(1))
        return [correct[:, :k].any(dim=1).float().mean().item() for k in topk]


@torch.no_grad()
def classifier_eval(
    feats:      list[torch.Tensor],
    labels:     list[torch.Tensor],
    classifier: nn.Module,
    chunk_size: int = 512,
) -> dict[str, float]:
    """Evaluate a linear classifier on pre-extracted features.

    Returns:
        {"val_loss": float, "val_acc1": float, "val_acc5": float,
         "feat_l1": float, "feat_l2": float}
    """
    classifier.eval()
    f = torch.cat(feats).float()
    l = torch.cat(labels)

    num_classes  = classifier(f[:1]).size(1)
    sum_loss = sum_acc1 = sum_acc5 = 0

    for f_chunk, l_chunk in zip(f.split(chunk_size), l.split(chunk_size)):
        logits     = classifier(f_chunk)
        b          = l_chunk.size(0)
        sum_loss  += F.cross_entropy(logits, l_chunk).item() * b
        acc1, acc5 = _topk_accuracy(logits, l_chunk, topk=(1, min(5, num_classes)))
        sum_acc1  += acc1 * b
        sum_acc5  += acc5 * b

    n = f.size(0)
    return {
        "val_loss": sum_loss / n,
        "val_acc1": sum_acc1 / n,
        "val_acc5": sum_acc5 / n,
    }


def feature_norms_eval(feats: torch.Tensor) -> dict[str, float]:
    """Average L1 and L2 norm across the batch."""
    return {
        "feat_l1": feats.norm(p=1, dim=1).mean().item(),
        "feat_l2": feats.norm(p=2, dim=1).mean().item(),
    }


@torch.no_grad()
def sparsity_eval(feats: list[torch.Tensor]) -> dict[str, float]:
    """Compute sparsity metrics on pre-extracted features.

    Returns:
        {"hoyer": float, "zero_pct": float}
    """
    f = torch.cat(feats).float()
    n  = f.size(1)
    l1 = f.norm(p=1, dim=1)
    l2 = f.norm(p=2, dim=1)
    hoyer = ((n ** 0.5 - l1 / l2.clamp(min=1e-8)) / (n ** 0.5 - 1)).mean().item()
    return {
        "hoyer":    hoyer,
        "zero_pct": (f.abs() <= 1e-5).float().mean().item(),
    }


@torch.no_grad()
def orthogonality_eval(feats: list[torch.Tensor], min_sum_frac: float = 0.01) -> dict[str, float]:
    """Measures how orthogonal the feature dimensions are.

    Column-normalizes features, computes the D×D correlation matrix, and
    measures deviation from the identity via off-diagonal absolute values.
    Lower is better — 0 means perfectly orthogonal dimensions.

    Returns:
        {"ortho_mean": float, "ortho_median": float}
    """
    f      = torch.cat(feats).float()
    active = f.sum(0) > min_sum_frac * f.size(0)
    f      = f[:, active]
    f      = F.normalize(f, dim=0)
    corr   = f.T @ f
    err    = (corr - torch.eye(corr.size(0), device=corr.device)).abs()
    mask   = ~torch.eye(corr.size(0), dtype=torch.bool, device=corr.device)
    off_diag = err[mask]
    return {
        "ortho_mean":   off_diag.mean().item(),
        "ortho_median": off_diag.median().item(),
    }


@torch.no_grad()
def mig_eval(
    feats:  list[torch.Tensor],
    labels: list[torch.Tensor],
    n_neighbors: int = 3,
) -> dict[str, float]:
    """Mutual Information Gap (MIG).

    Estimates MI between each feature dimension and the class label using
    sklearn's k-NN based estimator (mutual_info_classif), then returns the
    normalized gap between the top-2 MI values:

        MIG = (MI_top1 - MI_top2) / H(y)

    Returns:
        {"mig": float}
    """
    from sklearn.feature_selection import mutual_info_classif

    f = torch.cat(feats).float().cpu().numpy()  # (N, D)
    y = torch.cat(labels).cpu().numpy()          # (N,)

    # H(y) — exact, y is discrete
    counts = torch.bincount(torch.from_numpy(y).long())
    p_y    = counts.float() / counts.sum()
    H_y    = -(p_y * p_y.clamp(min=1e-10).log()).sum().item()

    mi = mutual_info_classif(f, y, n_neighbors=n_neighbors)  # (D,)

    top2 = sorted(mi)[-2:]
    mig  = (top2[1] - top2[0]) / H_y if len(top2) >= 2 else 0.0
    return {"mig": mig}


def evaluate_features(
    train_feats:  list[torch.Tensor],
    train_labels: list[torch.Tensor],
    val_feats:    list[torch.Tensor],
    val_labels:   list[torch.Tensor],
    classifier:   nn.Module,
    k: int = 20,
    temperature: float = 0.07,
    chunk_size: int = 512,
) -> dict[str, float]:
    """KNN + classifier + sparsity + norms + MIG + ortho."""
    knn   = knn_eval(train_feats, train_labels, val_feats, val_labels, k=k, temperature=temperature, chunk_size=chunk_size)
    clf   = classifier_eval(val_feats, val_labels, classifier, chunk_size=chunk_size)
    spa   = sparsity_eval(val_feats)
    nrm   = feature_norms_eval(torch.cat(val_feats).float())
    mig   = mig_eval(val_feats, val_labels)
    ortho = orthogonality_eval(val_feats)
    return {**knn, **clf, **spa, **nrm, **mig, **ortho}


def evaluate_features_fast(
    train_feats:  list[torch.Tensor],
    train_labels: list[torch.Tensor],
    val_feats:    list[torch.Tensor],
    val_labels:   list[torch.Tensor],
    classifier:   nn.Module,
    k: int = 20,
    temperature: float = 0.07,
    chunk_size: int = 512,
) -> dict[str, float]:
    """KNN + classifier only — no MIG, ortho, sparsity, or norms."""
    knn = knn_eval(train_feats, train_labels, val_feats, val_labels, k=k, temperature=temperature, chunk_size=chunk_size)
    clf = classifier_eval(val_feats, val_labels, classifier, chunk_size=chunk_size)
    return {**knn, **clf}

