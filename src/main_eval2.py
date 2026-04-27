"""Offline evaluation v2 — feature extraction and metric computation.

Forwards the val set once and saves encoder (and projector) features to disk,
then computes metrics from those features. Re-runs from cached features if the
npz already exists. Only --name is required; everything else comes from
saves/<name>/hparams.json.
"""

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import load_dataset
from losses import NTXentLoss
from model import SimCLRModel


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_features(model: SimCLRModel, loader, device: torch.device, projector_type: str) -> dict:
    """Forward the entire loader and collect encoder, proj_hidden, and proj activations.

    Returns:
        dict with keys:
            "encoder"     (N, 512)
            "proj_hidden" (N, proj_hidden_dim)  — only if projector_type != "none"
            "proj"        (N, proj_dim)          — only if projector_type != "none"
            "labels"      (N,)
    """
    is_cuda       = device.type == "cuda"
    has_projector = projector_type != "none"
    # mlp:    [Linear, ReLU, Linear]          → [:2] gives pre-output hidden
    # mlp-bn: [Linear, BN, ReLU, Linear, BN]  → [:3] gives pre-output hidden
    hidden_slice  = 2 if projector_type == "mlp" else 3

    enc_list, hidden_list, proj_list, label_list = [], [], [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=is_cuda)
        enc    = model.encode(images)
        enc_list.append(enc.cpu())
        label_list.append(labels)
        if has_projector:
            hidden_list.append(model.projector[:hidden_slice](enc).cpu())
            proj_list.append(model.projector(enc).cpu())

    result: dict = {
        "encoder": torch.cat(enc_list).numpy(),
        "labels":  torch.cat(label_list).numpy(),
    }
    if has_projector:
        result["proj_hidden"] = torch.cat(hidden_list).numpy()
        result["proj"]        = torch.cat(proj_list).numpy()
    return result


@torch.no_grad()
def _extract_with_labels(
    encode_fn: Callable[[torch.Tensor], torch.Tensor],
    loader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract (features, labels) from a loader using any encode_fn.

    Args:
        encode_fn: maps a batch of images already on device → (B, D) tensor
    """
    is_cuda = device.type == "cuda"
    feat_list, label_list = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=is_cuda)
        feat_list.append(encode_fn(images).cpu())
        label_list.append(labels)
    return torch.cat(feat_list).numpy(), torch.cat(label_list).numpy()



# ---------------------------------------------------------------------------
# Sparsity metrics
# ---------------------------------------------------------------------------

def sparsity(features: np.ndarray) -> dict:
    """Hoyer sparsity and near-zero activation percentage.

    Args:
        features: (N, D) float array

    Returns:
        dict with "hoyer" (scalar ∈ [0, 1]) and "zero_pct" (scalar ∈ [0, 1])
    """
    N, D = features.shape
    l1 = np.abs(features).sum(axis=1)           # (N,)
    l2 = np.sqrt((features ** 2).sum(axis=1))   # (N,)
    hoyer    = ((D ** 0.5 - l1 / np.maximum(l2, 1e-8)) / (D ** 0.5 - 1)).mean()
    zero_pct = (np.abs(features) < 1e-5).mean()
    return {"hoyer": float(hoyer), "zero_pct": float(zero_pct)}


# ---------------------------------------------------------------------------
# Class consistency rate
# ---------------------------------------------------------------------------

def class_consistency(
    features:  np.ndarray,
    labels:    np.ndarray,
    threshold: float = 1e-5,
) -> dict:
    """Proportion of activated samples belonging to the most frequent class, averaged over dims.

    For each dimension d: among all samples where feature_d > threshold,
    what fraction share the single most common class label?

    Args:
        features:  (N, D)
        labels:    (N,) integer class indices
        threshold: activation threshold (default 0 — positive activations only)

    Returns:
        dict with "class_consistency" scalar ∈ [0, 1]
    """
    N, D        = features.shape
    num_classes = int(labels.max()) + 1
    active      = (features > threshold)         # (N, D)  bool
    n_active    = active.sum(axis=0)             # (D,)    int

    # count_per_class[c, d] = # class-c samples active in dim d
    # Computed as one-hot(labels)ᵀ @ active — a single matmul
    one_hot           = np.zeros((N, num_classes), dtype=np.float32)
    one_hot[np.arange(N), labels] = 1.0
    count_per_class   = one_hot.T @ active.astype(np.float32)   # (C, D)

    most_frequent = count_per_class.max(axis=0)                  # (D,)
    rates         = most_frequent / np.maximum(n_active, 1)      # (D,)

    return {"class_consistency": float(rates.mean())}


# ---------------------------------------------------------------------------
# Dimensional correlation matrix
# ---------------------------------------------------------------------------

def dim_correlation(
    features: np.ndarray,
    n_dims:   int = 20,
    seed:     int = 0,
) -> dict:
    """Correlation matrix over a random subset of feature dimensions.

    C_{ij} = sum_x f̃_i(x) f̃_j(x)  where  f̃_i(x) = f_i(x) / ||f_i||_2 (across dataset).

    Returns scalars and the matrix itself (saved to npz, skipped in console summary):
        "dim_corr_matrix"        (n_dims, n_dims) — full matrix
        "dim_corr_indices"       (n_dims,)        — which dims were sampled
        "dim_corr_mean_offdiag"  scalar            — mean |C_ij| for i≠j
    """
    N, D  = features.shape
    rng   = np.random.default_rng(seed)
    idx   = rng.choice(D, min(n_dims, D), replace=False)

    F      = features[:, idx]                        # (N, n_dims)
    norms  = np.sqrt((F ** 2).sum(axis=0))           # (n_dims,)
    F_norm = F / np.maximum(norms, 1e-8)             # (N, n_dims)
    C      = F_norm.T @ F_norm                       # (n_dims, n_dims)

    offdiag_mask     = ~np.eye(C.shape[0], dtype=bool)
    mean_offdiag     = float(np.abs(C[offdiag_mask]).mean())

    return {
        "dim_corr_matrix":       C,
        "dim_corr_indices":      idx,
        "dim_corr_mean_offdiag": mean_offdiag,
    }


# ---------------------------------------------------------------------------
# Feature selection — EA and subset helpers
# ---------------------------------------------------------------------------

def compute_ea(train_features: np.ndarray) -> np.ndarray:
    """Expected Activation: EA_i = E_x[f̃_i(x)]  where f̃(x) = f(x)/‖f(x)‖_2.

    Args:
        train_features: (N, D)

    Returns:
        ea: (D,) mean activation per dimension over the L2-normalised train set
    """
    norms      = np.linalg.norm(train_features, axis=1, keepdims=True)  # (N, 1)
    normalised = train_features / np.maximum(norms, 1e-8)               # (N, D)
    return normalised.mean(axis=0)                                       # (D,)


# ---------------------------------------------------------------------------
# Linear probe (trained on pre-extracted features)
# ---------------------------------------------------------------------------

def _train_linear_probe(
    train_feats:  np.ndarray,
    train_labels: np.ndarray,
    num_classes:  int,
    epochs:       int = 100,
    lr:           float = 0.1,
    batch_size:   int = 1024,
    device:       torch.device = torch.device("cpu"),
) -> nn.Linear:
    D   = train_feats.shape[1]
    ds  = torch.utils.data.TensorDataset(
        torch.from_numpy(train_feats).float(),
        torch.from_numpy(train_labels).long(),
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
    clf = nn.Linear(D, num_classes).to(device)
    opt = torch.optim.SGD(clf.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    clf.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            F.cross_entropy(clf(xb), yb).backward()
            opt.step()
            opt.zero_grad()
    return clf


@torch.no_grad()
def _eval_accuracy(clf: nn.Linear, feats: np.ndarray, labels: np.ndarray) -> float:
    clf.eval()
    X = torch.from_numpy(feats).float().to(next(clf.parameters()).device)
    y = torch.from_numpy(labels).long().to(X.device)
    return (clf(X).argmax(dim=1) == y).float().mean().item()


def _eval_map_at_k(feats: np.ndarray, labels: np.ndarray, k: int = 10) -> float:
    """mAP@k via val-to-val L2-normalised cosine retrieval (self excluded)."""
    N = feats.shape[0]
    norms   = np.linalg.norm(feats, axis=1, keepdims=True)
    feats_n = feats / np.maximum(norms, 1e-8)

    ap_sum  = 0.0
    chunk   = 512
    for start in range(0, N, chunk):
        end  = min(start + chunk, N)
        sims = feats_n[start:end] @ feats_n.T               # (chunk, N)
        # Exclude self
        for i in range(end - start):
            sims[i, start + i] = -np.inf
        # top-k sorted descending (cheapest: full argsort on small N)
        top_idx = np.argsort(sims, axis=1)[:, -k:][:, ::-1] # (chunk, k)

        for i in range(end - start):
            ql    = labels[start + i]
            n_rel = int((labels == ql).sum()) - 1            # exclude self
            if n_rel == 0:
                continue
            hits  = (labels[top_idx[i]] == ql).astype(float)
            prec  = np.cumsum(hits) / (np.arange(k) + 1)
            ap_sum += (prec * hits).sum() / min(k, n_rel)

    return ap_sum / N


# ---------------------------------------------------------------------------
# Feature-subset evaluation
# ---------------------------------------------------------------------------

def evaluate_feature_subsets(
    encode_fn:    Callable[[torch.Tensor], torch.Tensor],
    train_loader,
    val_loader,
    num_classes:  int,
    device:       torch.device,
    n_select:     int = 128,
    probe_epochs: int = 100,
    probe_lr:     float = 0.1,
    seed:         int = 0,
) -> dict:
    """Evaluate 4 feature subsets via linear probe accuracy and mAP@10.

    Subsets:
        all              — all D dimensions
        rand{n}          — n random dimensions
        ea{n}            — n highest Expected-Activation dimensions
        ea{n}relu        — same as ea{n} but with ReLU applied after selection

    Args:
        encode_fn: maps images (already on device) → (B, D) tensor; controls
                   which model stage is evaluated (encoder, projector, …)

    Returns:
        flat dict with keys "<subset>_acc" and "<subset>_map10"
    """
    print("  extracting train features...", end=" ", flush=True)
    train_feats, train_labels = _extract_with_labels(encode_fn, train_loader, device)
    print("done")
    print("  extracting val features...", end=" ", flush=True)
    val_feats, val_labels = _extract_with_labels(encode_fn, val_loader, device)
    print("done")

    D   = train_feats.shape[1]
    ea  = compute_ea(train_feats)
    top_idx  = np.argsort(ea)[-n_select:]

    rng      = np.random.default_rng(seed)
    rand_idx = rng.choice(D, n_select, replace=False)

    n = n_select
    subsets = [
        ("all",          np.arange(D), False),
        (f"rand{n}",     rand_idx,     False),
        (f"ea{n}",       top_idx,      False),
        (f"ea{n}relu",   top_idx,      True),
    ]

    results: dict = {}
    for name, idx, use_relu in subsets:
        tr = train_feats[:, idx].copy()
        vl = val_feats[:, idx].copy()
        if use_relu:
            np.clip(tr, 0, None, out=tr)
            np.clip(vl, 0, None, out=vl)

        print(f"  [{name:12s}] training probe...", end=" ", flush=True)
        clf = _train_linear_probe(tr, train_labels, num_classes,
                                  epochs=probe_epochs, lr=probe_lr, device=device)
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

def _batched_nce(
    z1:         np.ndarray,
    z2:         np.ndarray,
    loss_fn:    NTXentLoss,
    batch_size: int,
    device:     torch.device,
) -> float:
    """Symmetric NTXent averaged over mini-batches of pre-extracted features."""
    total, n_batches = 0.0, 0
    for start in range(0, len(z1), batch_size):
        z1b = torch.from_numpy(z1[start:start + batch_size]).float().to(device)
        z2b = torch.from_numpy(z2[start:start + batch_size]).float().to(device)
        with torch.no_grad():
            total += loss_fn(z1b, z2b).item()
        n_batches += 1
    return total / n_batches


@torch.no_grad()
def compute_disentanglement(
    encode_fn:             Callable[[torch.Tensor], torch.Tensor],
    train_loader_two_view,
    device:                torch.device,
    temperature:           float = 0.1,
    nce_batch_size:        int   = 256,
) -> dict:
    """Compute SEPIN@1/10/100/all without redundant NCE evaluations.

    Algorithm:
      1. Extract two-view train features once via encode_fn.
      2. Rank all D features by Expected Activation.
      3. Compute NTXent(f) once.
      4. Compute NTXent(f_{!=i}) for every i — D evaluations total.
      5. delta_i = NTXent(f_{!=i}) - NTXent(f)  (unique MI of dim i)
      6. SEPIN@k = mean of delta values for the top-k EA-ranked features.

    Args:
        encode_fn: maps images (on device) → (B, D) tensor; selects model stage
    """
    is_cuda = device.type == "cuda"

    print("  extracting two-view train features...", end=" ", flush=True)
    z1_list, z2_list = [], []
    for x1, x2, _ in train_loader_two_view:
        z1_list.append(encode_fn(x1.to(device, non_blocking=is_cuda)).cpu())
        z2_list.append(encode_fn(x2.to(device, non_blocking=is_cuda)).cpu())
    z1 = torch.cat(z1_list).numpy()
    z2 = torch.cat(z2_list).numpy()
    print("done")

    D       = z1.shape[1]
    loss_fn = NTXentLoss(temperature)

    # EA ranking — computed once, used for all SEPIN@k
    ea         = compute_ea(z1)
    ea_rank    = np.argsort(ea)[::-1].copy()   # dim indices sorted highest EA first

    nce_full = _batched_nce(z1, z2, loss_fn, nce_batch_size, device)
    print(f"  NTXent full (D={D}): {nce_full:.4f}")

    # One leave-one-out evaluation per dimension
    all_dims = np.arange(D)
    deltas   = np.empty(D)            # deltas[i] = ΔNCE when dim i is removed
    for i in range(D):
        mask      = all_dims != i
        deltas[i] = _batched_nce(z1[:, mask], z2[:, mask], loss_fn, nce_batch_size, device) - nce_full

    # deltas ranked by EA (highest EA first)
    deltas_ranked = deltas[ea_rank]   # (D,) — position j = EA rank j

    results: dict = {"nce_full": nce_full}
    for k in (1, 10, 100, D):
        label = "all" if k == D else str(k)
        results[f"sepin_{label}"] = float(deltas_ranked[:k].mean())

    print(f"  SEPIN@1={results['sepin_1']:.4f}  @10={results['sepin_10']:.4f}"
          f"  @100={results['sepin_100']:.4f}  @all={results['sepin_all']:.4f}")

    return results


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_model_weights(path: Path, model: SimCLRModel) -> int:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    return ckpt["epoch"]


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Offline evaluation v2 for a saved SimCLR run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--name",          required=True,
                   help="Run name — points to saves/<name>/.")
    p.add_argument("--epoch",         default=None, type=int,
                   help="Checkpoint epoch to evaluate. Default: latest.")
    p.add_argument("--data-root",     default=None,
                   help="Override data root from hparams (useful on a different machine).")
    p.add_argument("--num-workers",   default=None, type=int,
                   help="Override num_workers from hparams.")
    p.add_argument("--n-select",      default=128,  type=int,
                   help="Number of features to select for rand/EA subsets.")
    p.add_argument("--probe-epochs",  default=100,  type=int)
    p.add_argument("--probe-lr",      default=0.1,  type=float)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args     = parse_args()
    save_dir = Path("saves") / args.name

    if not save_dir.exists():
        raise SystemExit(f"Run directory '{save_dir}' does not exist.")

    hparams = json.loads((save_dir / "hparams.json").read_text())
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Data ---
    dataset    = hparams["dataset"]
    image_size = 32 if dataset == "cifar100" else 64
    num_classes = 100 if dataset == "cifar100" else 200

    def get_loader(train: bool):
        return load_dataset(
            dataset, two_view=False, augment=None, train=train,
            batch_size=hparams["batch_size"],
            num_workers=hparams["num_workers"],
            data_root=hparams["data_root"],
        )

    val_loader        = get_loader(train=False)
    train_loader      = get_loader(train=True)
    train_loader_tv   = load_dataset(
        dataset, two_view=True, augment=hparams.get("method", "simclr"), train=True,
        batch_size=hparams["batch_size"],
        num_workers=hparams["num_workers"],
        data_root=hparams["data_root"],
    )

    # --- Model ---
    projector_type = hparams.get("projector", "mlp")
    has_projector  = projector_type != "none"

    model = SimCLRModel(
        proj_hidden=hparams["proj_hidden_dim"],
        proj_dim=hparams["proj_output_dim"],
        image_size=image_size,
        projector=projector_type,
        non_neg=hparams.get("non_neg", False),
    ).to(device)

    epoch = load_model_weights(ckpt_path, model)
    model.eval()

    # --- Val feature extraction ---
    features = extract_features(model, val_loader, device, projector_type)

    val_labels    = features["labels"].astype(int)
    hidden_slice  = 2 if projector_type == "mlp" else 3
    temperature   = hparams.get("temperature", 0.1)

    # Layers to evaluate: (name, val_features_key, encode_fn)
    layers = [("encoder", "encoder", lambda imgs: model.encode(imgs))]
    if has_projector:
        layers += [
            ("proj_hidden", "proj_hidden",
             lambda imgs, s=hidden_slice: model.projector[:s](model.encode(imgs))),
            ("proj", "proj",
             lambda imgs: model.projector(model.encode(imgs))),
        ]

    results: dict = {"epoch": epoch}

    for name, feat_key, encode_fn in layers:
        val_feats = features[feat_key]

        # --- Point metrics (sparsity, class consistency, dim correlation) ---
        m = {
            **sparsity(val_feats),
            **class_consistency(val_feats, val_labels),
            **dim_correlation(val_feats),
        }
        results.update({f"{name}_{k}": v for k, v in m.items()})

        # --- Feature-subset evaluation ---
        print(f"\n--- Feature subsets: {name} ---")
        sub = evaluate_feature_subsets(
            encode_fn, train_loader, val_loader, num_classes, device,
            n_select=args.n_select, probe_epochs=args.probe_epochs, probe_lr=args.probe_lr,
        )
        results.update({f"{name}_{k}": v for k, v in sub.items()})

        # --- Disentanglement (SEPIN) ---
        print(f"\n--- Disentanglement: {name} ---")
        dis = compute_disentanglement(
            encode_fn, train_loader_tv, device,
            temperature=temperature, nce_batch_size=hparams["batch_size"],
        )
        results.update({f"{name}_{k}": v for k, v in dis.items()})

    # --- Print summary ---
    col_w = max(len(k) for k in results) + 2
    print(f"\n{'Metric':<{col_w}} Value")
    print("-" * (col_w + 10))
    for k, v in results.items():
        if isinstance(v, np.ndarray):
            continue
        print(f"{k:<{col_w}} {v:.4f}" if isinstance(v, float) else f"{k:<{col_w}} {v}")

    # --- Save metrics ---
    metrics_path = save_dir / "eval2_metrics" / f"metrics_{epoch:04d}.npz"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(metrics_path, **{k: np.array(v) for k, v in results.items()})
    print(f"\nSaved metrics → {metrics_path}")


if __name__ == "__main__":
    main()
