"""
Analyze and compare training runs.

Usage:
    python analyze.py TestL0                 # saves plots to saves/TestL0/plots/
    python analyze.py TestL0 TestL1 TestL2   # compare multiple runs, saves to saves/TestL0/plots/
    python analyze.py TestL0 --show          # show interactive plots instead of saving
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

SAVES_DIR = Path("saves")
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def load_run(name: str) -> tuple[dict, pd.DataFrame]:
    run_dir = SAVES_DIR / name
    with open(run_dir / "hparams.json") as f:
        hparams = json.load(f)
    df = pd.read_csv(run_dir / "metrics.csv")
    return hparams, df


def label(hparams: dict) -> str:
    return hparams.get("name", hparams.get("proj_layers", "?"))


# Metrics shown per layer, in row-major order for a 3×5 grid.
_LAYER_METRICS = [
    ("grad",         "Gradient Norm"),
    ("train_acc1",   "Train Acc@1"),
    ("knn_acc1",     "KNN Acc@1"),
    ("knn_acc5",     "KNN Acc@5"),
    ("val_loss",     "Val Loss"),
    ("val_acc1",     "Val Acc@1"),
    ("val_acc5",     "Val Acc@5"),
    ("hoyer",        "Hoyer"),
    ("zero_pct",     "Zero %"),
    ("feat_l1",      "Feat L1"),
    ("feat_l2",      "Feat L2"),
    ("mig",          "MIG"),
    ("ortho_mean",   "Ortho Mean"),
    ("ortho_median", "Ortho Median"),
]


def _feature_names_from_hparams(hp: dict) -> list[str]:
    """Reconstruct layer names from hparams (mirrors model.feature_names)."""
    n = hp.get("proj_layers", 0)
    if hp.get("no_projector", False):
        return ["backbone"]
    return ["backbone"] + [f"proj_{i}" for i in range(n)] + ["head"]


def _all_layers(runs: list[tuple[str, dict, pd.DataFrame]]) -> list[str]:
    """Union of layer names across all runs, preserving depth order."""
    seen: dict[str, None] = {}
    for _, hp, _ in runs:
        for name in _feature_names_from_hparams(hp):
            seen[name] = None
    # Sort: backbone first, then proj_0..N, then head
    def sort_key(n):
        if n == "backbone": return (0, 0)
        if n == "head":     return (2, 0)
        return (1, int(n.split("_")[1]))
    return sorted(seen, key=sort_key)


def _plot_ax(ax, runs, col: str, title: str):
    for color, (_, hp, df) in zip(COLORS, runs):
        if col not in df.columns:
            continue
        valid = df[["epoch", col]].dropna()
        if valid.empty:
            continue
        ax.plot(valid["epoch"], valid[col], label=label(hp), color=color, linewidth=1.5)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    ax.ticklabel_format(useOffset=False, axis="y")


def make_figures(runs: list[tuple[str, dict, pd.DataFrame]], save_dir: Path | None):
    figs = []

    # ── Figure 1: Run-level training metrics ─────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("Training", fontsize=12)
    for ax, col, title in zip(axes, ["train_nce_loss", "train_cls_loss", "lr"], ["NCE Loss", "Classifier Loss", "LR"]):
        _plot_ax(ax, runs, col, title)
    fig.tight_layout()
    figs.append(("training", fig))

    # ── One figure per layer ──────────────────────────────────────────────
    n_metrics = len(_LAYER_METRICS)
    n_cols = 5
    n_rows = -(-n_metrics // n_cols)  # ceil division

    for layer in _all_layers(runs):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.2, n_rows * 3.2))
        fig.suptitle(f"Layer: {layer}", fontsize=13)
        axes_flat = axes.flatten()
        for ax, (suffix, title) in zip(axes_flat, _LAYER_METRICS):
            _plot_ax(ax, runs, f"{layer}_{suffix}", title)
        # Hide any unused subplots
        for ax in axes_flat[n_metrics:]:
            ax.set_visible(False)
        fig.tight_layout()
        figs.append((layer, fig))

    # ── Save or show ──────────────────────────────────────────────────────
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        for name, fig in figs:
            path = save_dir / f"{name}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved {path}")
        plt.close("all")
    else:
        plt.show()


def print_summary(runs: list[tuple[str, dict, pd.DataFrame]]):
    print(f"\n{'Run':<12} {'proj_layers':<12} {'Val Acc@1 (backbone)':<22} {'KNN Acc@1 (backbone)':<22} {'Val Acc@1 (head)'}")
    print("-" * 90)
    for name, hp, df in runs:
        def best(col):
            valid = df[col].dropna()
            return f"{valid.max():.4f}" if not valid.empty else "N/A"

        print(
            f"{name:<12} {hp.get('proj_layers', '?'):<12} "
            f"{best('backbone_val_acc1'):<22} {best('backbone_knn_acc1'):<22} "
            f"{best('head_val_acc1')}"
        )
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--show", action="store_true", help="Show interactive plots instead of saving")
    args = parser.parse_args()

    runs = []
    for name in args.runs:
        try:
            hp, df = load_run(name)
            runs.append((name, hp, df))
            print(f"Loaded {name}: {len(df)} epochs, proj_layers={hp.get('proj_layers')}")
        except FileNotFoundError as e:
            print(f"Warning: {e}")

    if not runs:
        print("No runs found.")
        return

    print_summary(runs)
    save_dir = None if args.show else SAVES_DIR / args.runs[0] / "plots"
    make_figures(runs, save_dir)


if __name__ == "__main__":
    main()
