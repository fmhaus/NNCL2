"""Compare metrics.csv from two training runs on the same plots."""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def _add_series(ax,epochs1, s1, epochs2, s2, label, color, name1, name2):
    """Plot one metric series for both runs with shared color, distinct style."""
    ax.plot(epochs1, s1, label=f"{label} ({name1})", color=color, linewidth=1.5)
    ax.plot(epochs2, s2, label=f"{label} ({name2})", color=color, linewidth=1.5,
            linestyle="--", alpha=0.5)


def compare(csv1: Path, csv2: Path, name1: str, name2: str, out_path: Path | None = None) -> None:
    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2)
    e1, e2 = df1["epoch"], df2["epoch"]

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        f"{name1}  vs  {name2}\n"
        f"solid = {name1} · dashed/faded = {name2}",
        fontsize=12, fontweight="bold",
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    def color(i):
        return prop_cycle[i % len(prop_cycle)]

    n1, n2 = name1, name2

    # --- Loss ---
    ax = fig.add_subplot(gs[0, 0])
    _add_series(ax, e1, df1["train_nce_loss"],   e2, df2["train_nce_loss"],   "NCE (train)",  color(0), n1, n2)
    _add_series(ax, e1, df1["train_class_loss"], e2, df2["train_class_loss"], "Cls (train)",  color(1), n1, n2)
    _add_series(ax, e1, df1["val_loss"],         e2, df2["val_loss"],         "Cls (val)",    color(2), n1, n2)
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.legend(fontsize=7)

    # --- Top-1 accuracy ---
    ax = fig.add_subplot(gs[0, 1])
    _add_series(ax, e1, df1["train_acc1_epoch"], e2, df2["train_acc1_epoch"], "Linear (train)", color(0), n1, n2)
    _add_series(ax, e1, df1["val_acc1"],         e2, df2["val_acc1"],         "Linear (val)",   color(1), n1, n2)
    _add_series(ax, e1, df1["val_knn_acc1"],     e2, df2["val_knn_acc1"],     "kNN (val)",      color(2), n1, n2)
    ax.set_title("Top-1 Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend(fontsize=7)

    # --- Top-5 accuracy ---
    ax = fig.add_subplot(gs[0, 2])
    _add_series(ax, e1, df1["train_acc5_epoch"], e2, df2["train_acc5_epoch"], "Linear (train)", color(0), n1, n2)
    _add_series(ax, e1, df1["val_acc5"],         e2, df2["val_acc5"],         "Linear (val)",   color(1), n1, n2)
    _add_series(ax, e1, df1["val_knn_acc5"],     e2, df2["val_knn_acc5"],     "kNN (val)",      color(2), n1, n2)
    ax.set_title("Top-5 Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend(fontsize=7)

    # --- Learning rate ---
    ax = fig.add_subplot(gs[1, 0])
    _add_series(ax, e1, df1["lr"], e2, df2["lr"], "LR", color(0), n1, n2)
    ax.set_title("Learning Rate")
    ax.set_xlabel("Epoch")
    ax.legend(fontsize=7)

    # --- Epoch time ---
    ax = fig.add_subplot(gs[1, 1])
    _add_series(ax, e1, df1["epoch_time_s"], e2, df2["epoch_time_s"], "Epoch time", color(0), n1, n2)
    ax.set_title("Epoch Time (s)")
    ax.set_xlabel("Epoch")
    ax.legend(fontsize=7)

    # --- Wall time ---
    ax = fig.add_subplot(gs[1, 2])
    _add_series(ax, e1, df1["wall_time"] / 3600, e2, df2["wall_time"] / 3600, "Wall time", color(0), n1, n2)
    ax.set_title("Wall Time (h)")
    ax.set_xlabel("Epoch")
    ax.legend(fontsize=7)

    if out_path is not None:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {out_path}")
    else:
        plt.show()


def main():
    p = argparse.ArgumentParser(description="Compare metrics.csv from two training runs.")
    p.add_argument("--name1", required=True, help="First run name  — resolves to saves/<name>/metrics.csv.")
    p.add_argument("--name2", required=True, help="Second run name — resolves to saves/<name>/metrics.csv.")
    p.add_argument("--out",   default=None,  help="Save figure to this path instead of showing it.")
    args = p.parse_args()

    def resolve(name):
        path = Path("saves") / name / "metrics.csv"
        if not path.exists():
            raise SystemExit(f"metrics.csv not found at '{path}'.")
        return path

    compare(resolve(args.name1), resolve(args.name2), args.name1, args.name2,
            Path(args.out) if args.out else None)


if __name__ == "__main__":
    main()
