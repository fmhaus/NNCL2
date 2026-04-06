"""Visualize metrics.csv for one or more training runs."""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Line styles cycled per run; colors are per metric series (shared across runs).
_STYLES = ["-", "--", ":", "-."]
_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def _color(i: int) -> str:
    return _COLORS[i % len(_COLORS)]


def _style(run_idx: int) -> str:
    return _STYLES[run_idx % len(_STYLES)]


def _plot_series(ax, runs: list[tuple[str, pd.DataFrame]], columns: list[str], labels: list[str]) -> None:
    """Plot multiple metric columns across multiple runs.

    Each column gets a fixed color; each run gets a fixed linestyle.
    """
    for col_idx, (col, label) in enumerate(zip(columns, labels)):
        for run_idx, (name, df) in enumerate(runs):
            if col not in df.columns:
                continue
            ax.plot(
                df["epoch"], df[col],
                color=_color(col_idx),
                linestyle=_style(run_idx),
                linewidth=1.4,
                label=f"{label} ({name})" if len(runs) == 1 else None,
            )


def _add_run_legend(ax, runs: list[tuple[str, pd.DataFrame]]) -> None:
    """Add a run→linestyle legend inside a given axes."""
    handles = [
        mlines.Line2D([], [], color="black", linestyle=_style(i), linewidth=1.4, label=name)
        for i, (name, _) in enumerate(runs)
    ]
    ax.legend(handles=handles, title="Run", fontsize=7, title_fontsize=7, loc="best")


def plot(runs: list[tuple[str, pd.DataFrame]], out_path: Path | None = None) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle(" · ".join(n for n, _ in runs), fontsize=11, fontweight="bold")
    axes = axes.flatten()

    panels = [
        # (title, columns, labels)
        ("Loss",             ["train_nce_loss", "train_class_loss", "val_loss"],      ["NCE (train)", "Cls (train)", "Cls (val)"]),
        ("Top-1 Accuracy",   ["train_acc1_epoch", "val_acc1", "val_knn_acc1"],        ["Linear (train)", "Linear (val)", "kNN (val)"]),
        ("Top-5 Accuracy",   ["train_acc5_epoch", "val_acc5", "val_knn_acc5"],        ["Linear (train)", "Linear (val)", "kNN (val)"]),
        ("Feat L1 norm",     ["feat_l1"],                                              ["L1"]),
        ("Grad: backbone",   ["grad_backbone_norm"],                                   ["backbone"]),
        ("Grad: feat",       ["grad_feat_norm"],                                       ["feat"]),
        ("Grad: proj out",   ["grad_proj_out_norm"],                                   ["proj_out"]),
        ("Feat L2 norm",     ["feat_l2"],                                              ["L2"]),
    ]

    for ax, (title, columns, labels) in zip(axes, panels):
        _plot_series(ax, runs, columns, labels)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.tick_params(labelsize=7)
        if len(runs) == 1:
            ax.legend(fontsize=7)

    if len(runs) > 1:
        # Run→style legend in the last panel (feat_l2)
        _add_run_legend(axes[-1], runs)

    plt.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {out_path}")
    else:
        plt.show()


def main() -> None:
    p = argparse.ArgumentParser(description="Visualize metrics from one or more training runs.")
    p.add_argument("--names", nargs="+", required=True,
                   help="One or more run names. Resolves to saves/<name>/metrics.csv.")
    p.add_argument("--out", default=None, help="Save figure to this path instead of showing.")
    args = p.parse_args()

    runs: list[tuple[str, pd.DataFrame]] = []
    for name in args.names:
        csv = Path("saves") / name / "metrics.csv"
        if not csv.exists():
            raise SystemExit(f"metrics.csv not found at '{csv}'.")
        runs.append((name, pd.read_csv(csv)))

    plot(runs, Path(args.out) if args.out else None)


if __name__ == "__main__":
    main()
