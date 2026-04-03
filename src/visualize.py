"""Visualize metrics.csv produced by a training run."""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot(csv_path: Path, out_path: Path | None = None) -> None:
    df = pd.read_csv(csv_path)
    epochs = df["epoch"]

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(csv_path.parent.name, fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # --- Loss ---
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(epochs, df["train_nce_loss"],   label="NCE (train)")
    ax.plot(epochs, df["train_class_loss"], label="Cls (train)")
    ax.plot(epochs, df["val_loss"],         label="Cls (val)", linestyle="--")
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.legend(fontsize=8)

    # --- Top-1 accuracy ---
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(epochs, df["train_acc1_epoch"], label="Linear (train)")
    ax.plot(epochs, df["val_acc1"],         label="Linear (val)", linestyle="--")
    ax.plot(epochs, df["val_knn_acc1"],     label="kNN (val)",    linestyle=":")
    ax.set_title("Top-1 Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend(fontsize=8)

    # --- Top-5 accuracy ---
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(epochs, df["train_acc5_epoch"], label="Linear (train)")
    ax.plot(epochs, df["val_acc5"],         label="Linear (val)", linestyle="--")
    ax.plot(epochs, df["val_knn_acc5"],     label="kNN (val)",    linestyle=":")
    ax.set_title("Top-5 Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend(fontsize=8)

    # --- Learning rate ---
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(epochs, df["lr"])
    ax.set_title("Learning Rate")
    ax.set_xlabel("Epoch")

    # --- Epoch time ---
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(epochs, df["epoch_time_s"])
    ax.set_title("Epoch Time (s)")
    ax.set_xlabel("Epoch")

    # --- Wall time ---
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(epochs, df["wall_time"] / 3600)
    ax.set_title("Wall Time (h)")
    ax.set_xlabel("Epoch")

    if out_path is not None:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {out_path}")
    else:
        plt.show()


def main():
    p = argparse.ArgumentParser(description="Plot metrics.csv from a training run.")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--name", help="Run name — resolves to saves/<name>/metrics.csv.")
    group.add_argument("--run",  help="Path to run directory or directly to metrics.csv.")
    p.add_argument("--out", default=None, help="Save figure to this path instead of showing it.")
    args = p.parse_args()

    path = Path("saves") / args.name if args.name else Path(args.run)
    csv_path = path if path.suffix == ".csv" else path / "metrics.csv"
    if not csv_path.exists():
        raise SystemExit(f"metrics.csv not found at '{csv_path}'.")

    out_path = Path(args.out) if args.out else None
    plot(csv_path, out_path)


if __name__ == "__main__":
    main()
