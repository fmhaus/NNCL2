"""Visualize evaluation results from main_eval2.py NPZ files.

Layout (one column per layer):
  Row 0 — scalar metrics (hoyer, zero_pct, class_consistency, dim_corr_offdiag)
  Row 1 — linear probe acc & mAP@10 by feature subset
  Row 2 — SEPIN @ 1 / 10 / 100 / all
  Row 3 — dim correlation heatmap
"""

import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors


LAYER_ORDER = ["encoder", "proj_hidden", "proj"]
ACCENT      = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Visualize eval2 metrics from a saved NPZ file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--name",  default=None,
                   help="Run name — points to saves/<name>/eval2_metrics/.")
    p.add_argument("--epoch", default=None, type=int,
                   help="Epoch to load (used with --name). Default: latest.")
    p.add_argument("--path",  default=None, metavar="NPZ_PATH",
                   help="Direct path to a .npz file (overrides --name/--epoch).")
    p.add_argument("--label", default=None,
                   help="Override the title label shown in the figure.")
    p.add_argument("--save",  default=None, metavar="PATH",
                   help="Save figure to file instead of showing.")
    args = p.parse_args()
    if args.path is None and args.name is None:
        p.error("provide either --name or --path")
    return args


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(name: str, epoch: int | None):
    metrics_dir = Path("saves") / name / "eval2_metrics"
    if not metrics_dir.exists():
        raise SystemExit(f"Metrics directory '{metrics_dir}' does not exist.")
    npz_files = sorted(metrics_dir.glob("metrics_*.npz"))
    if not npz_files:
        raise SystemExit(f"No metrics files found in '{metrics_dir}'.")
    if epoch is not None:
        path = metrics_dir / f"metrics_{epoch:04d}.npz"
        if not path.exists():
            raise SystemExit(f"'{path}' not found.")
    else:
        path = npz_files[-1]
    print(f"Loading '{path}'")
    return np.load(path, allow_pickle=True), path


def detect_layers(data) -> list[str]:
    present = {
        m.group(1)
        for k in data.files
        if (m := re.match(r"^(encoder|proj_hidden|proj)_hoyer$", k))
    }
    return [l for l in LAYER_ORDER if l in present]


def detect_n_select(data, layer: str) -> int:
    for k in data.files:
        m = re.match(rf"^{layer}_ea(\d+)_acc$", k)
        if m:
            return int(m.group(1))
    return 128


def get(data, key: str) -> float | None:
    return float(data[key]) if key in data.files else None


# ---------------------------------------------------------------------------
# Individual subplot renderers
# ---------------------------------------------------------------------------

def plot_scalars(ax, data, layer: str) -> None:
    items = [
        ("hoyer",         f"{layer}_hoyer"),
        ("zero_pct",      f"{layer}_zero_pct"),
        ("dead_pct",      f"{layer}_dead_pct"),
        ("class_cons",    f"{layer}_class_consistency"),
    ]
    labels = [l for l, k in items if k in data.files]
    values = [float(data[k]) * 100 for _, k in items if k in data.files]

    y      = np.arange(len(labels))
    colors = [ACCENT[i % len(ACCENT)] for i in range(len(labels))]
    bars   = ax.barh(y, values, color=colors, height=0.55)
    ax.bar_label(bars, fmt="%.2f%%", padding=4, fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_title(f"{layer}", fontsize=10, fontweight="bold")
    ax.tick_params(axis="x", labelsize=7)
    ax.xaxis.grid(True, linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_xlabel("%", fontsize=7)


def plot_probe(ax, data, layer: str, n: int) -> None:
    subsets = ["all", f"rand{n}", f"ea{n}", f"ea{n}relu"]
    accs  = [v * 100 if (v := get(data, f"{layer}_{s}_acc"))   is not None else 0.0 for s in subsets]
    maps  = [v * 100 if (v := get(data, f"{layer}_{s}_map10")) is not None else 0.0 for s in subsets]

    x, w = np.arange(len(subsets)), 0.36
    ax.bar(x - w / 2, accs, w, label="acc %",    color=ACCENT[0], zorder=3)
    ax.bar(x + w / 2, maps, w, label="mAP@10 %", color=ACCENT[1], zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(subsets, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("%", fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.yaxis.grid(True, linewidth=0.4, alpha=0.5, zorder=0)
    ax.set_ylim(0, 100)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("probe acc & mAP@10", fontsize=9)


def plot_sepin(ax, data, layer: str, ylim_max: float = 0.005) -> None:
    labels = ["@1", "@10", "@100", "@all"]
    keys   = [f"{layer}_sepin_1", f"{layer}_sepin_10",
              f"{layer}_sepin_100", f"{layer}_sepin_all"]
    values = [get(data, k) or 0.0 for k in keys]

    x      = np.arange(len(labels))
    colors = [ACCENT[2] if v >= 0 else ACCENT[3] for v in values]
    ax.bar(x, values, color=colors, zorder=3)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylim(0, ylim_max)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.yaxis.grid(True, linewidth=0.4, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("SEPIN", fontsize=9)

    nce = get(data, f"{layer}_nce_full")
    if nce is not None:
        ax.text(0.98, 0.98, f"NTXent={nce:.4f}",
                transform=ax.transAxes, fontsize=7,
                ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))


def plot_corr(ax, data, layer: str) -> None:
    key = f"{layer}_dim_corr_matrix"
    if key not in data.files:
        ax.axis("off")
        ax.text(0.5, 0.5, "no matrix", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="grey")
        return
    _wbw = mcolors.LinearSegmentedColormap.from_list(
        "wbw", [(0.0, "white"), (0.5, "black"), (1.0, "white")]
    )
    C  = data[key]
    im = ax.imshow(C, aspect="auto", cmap=_wbw, vmin=-1, vmax=1, interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("dim corr matrix", fontsize=9)
    ax.tick_params(labelsize=7)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    if args.path:
        p = Path(args.path)
        if not p.exists():
            raise SystemExit(f"'{p}' not found.")
        print(f"Loading '{p}'")
        data  = np.load(p, allow_pickle=True)
        npz_path = p
        label = p.stem
    else:
        data, npz_path = load_data(args.name, args.epoch)
        label = args.name
    if args.label:
        label = args.label
    epoch  = int(data["epoch"]) if "epoch" in data.files else "?"
    layers = detect_layers(data)

    if not layers:
        raise SystemExit("No recognisable layer data found in this NPZ file.")

    n = len(layers)
    fig = plt.figure(figsize=(5.5 * n, 17))
    fig.suptitle(f"{label}  —  epoch {epoch}", fontsize=13, fontweight="bold", y=0.998)

    gs = gridspec.GridSpec(
        4, n, figure=fig,
        height_ratios=[1.1, 1.2, 1.0, 1.4],
        hspace=0.55, wspace=0.38,
        top=0.965, bottom=0.04, left=0.07, right=0.97,
    )

    sepin_keys = [f"{l}_sepin_{s}" for l in layers for s in ("1", "10", "100", "all")]
    sepin_max  = max((get(data, k) or 0.0 for k in sepin_keys), default=0.005)
    sepin_ylim = max(sepin_max * 1.1, 1e-9)

    for col, layer in enumerate(layers):
        n_sel = detect_n_select(data, layer)
        plot_scalars(fig.add_subplot(gs[0, col]), data, layer)
        plot_probe  (fig.add_subplot(gs[1, col]), data, layer, n_sel)
        plot_sepin  (fig.add_subplot(gs[2, col]), data, layer, sepin_ylim)
        plot_corr   (fig.add_subplot(gs[3, col]), data, layer)

    auto_save = npz_path.with_suffix(".png")
    fig.savefig(auto_save, dpi=150, bbox_inches="tight")
    print(f"Saved → {auto_save}")

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved → {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
