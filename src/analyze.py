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


def _layer_sort_key(n: str) -> tuple:
    l1 = n.endswith("_l1")
    base = n[:-3] if l1 else n
    if base == "backbone": return (0, 0, int(l1))
    if base == "head":     return (2, 0, int(l1))
    if base.startswith("proj_"):
        try: return (1, int(base.split("_")[1]), int(l1))
        except: pass
    return (3, 0, int(l1))


def _all_layers(runs: list[tuple[str, dict, pd.DataFrame]]) -> list[str]:
    """Discover layer names from CSV columns across all runs."""
    suffixes = {suffix for suffix, _ in _LAYER_METRICS}
    seen: dict[str, None] = {}
    for _, _, df in runs:
        for col in df.columns:
            for suffix in suffixes:
                if col.endswith(f"_{suffix}"):
                    seen[col[: -(len(suffix) + 1)]] = None
                    break
    return sorted(seen, key=_layer_sort_key)


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


def make_figures(runs: list[tuple[str, dict, pd.DataFrame]], save_dir: Path, show: bool = False):
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

    # ── Save and optionally show ──────────────────────────────────────────
    save_dir.mkdir(parents=True, exist_ok=True)
    for name, fig in figs:
        path = save_dir / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved {path}")
    if show:
        plt.show()
    plt.close("all")


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
    parser.add_argument("--show", action="store_true", help="Show interactive plots in addition to saving")
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
    save_dir = SAVES_DIR / args.runs[0] / "plots"
    make_figures(runs, save_dir, show=args.show)


if __name__ == "__main__":
    main()
