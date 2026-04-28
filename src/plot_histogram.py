"""Plot activation histograms from saves/<name>/histograms/<layer>_0200.npz.

Usage:
    python plot_histogram.py --names BaseMLP BaseMLPBN --layer encoder_out
    python plot_histogram.py --names BaseMLP BaseMLPBN --layer encoder_out --separate
    python plot_histogram.py --names BaseMLP BaseMLPBN --layer encoder_out --out hist.png
"""

import argparse
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

SAVES_DIR = Path(__file__).parent.parent / "saves"


def load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(path)
    return d["counts"].astype(float), d["edges"]


def run_label(path: Path) -> str:
    name_file = path.parents[1] / "name.txt"
    if name_file.exists():
        return name_file.read_text().strip()
    return path.parents[1].name


def plot_overlay(paths: list[Path], ax: plt.Axes) -> None:
    for p in paths:
        counts, edges = load(p)
        density = counts / (counts.sum() * (edges[1] - edges[0]))
        ax.stairs(density, edges, label=run_label(p), linewidth=1.2, fill=False)
    ax.set_xlabel("Activation value")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)


def plot_separate(paths: list[Path], axes: list[plt.Axes]) -> None:
    for p, ax in zip(paths, axes):
        counts, edges = load(p)
        density = counts / (counts.sum() * (edges[1] - edges[0]))
        ax.stairs(density, edges, linewidth=1.0, fill=True, alpha=0.6)
        ax.set_title(run_label(p), fontsize=9)
        ax.set_xlabel("Activation value")
        ax.set_ylabel("Density")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--names", nargs="+", metavar="NAME",
                        help="Run names (default: all under saves/)")
    parser.add_argument("--layer", required=True, metavar="LAYER",
                        help="Layer name, e.g. encoder_out, proj_hidden, proj_out")
    parser.add_argument("--separate", action="store_true",
                        help="One subplot per run instead of overlaid")
    parser.add_argument("--out", type=Path, default=None,
                        help="Save to file instead of showing")
    args = parser.parse_args()

    if args.names:
        names = args.names
    else:
        names = [p.name for p in sorted(SAVES_DIR.iterdir()) if p.is_dir()]

    paths = []
    for name in names:
        p = SAVES_DIR / name / "histograms" / f"{args.layer}_0200.npz"
        if p.exists():
            paths.append(p)
        else:
            print(f"WARNING: {p} not found, skipping")

    if not paths:
        raise FileNotFoundError("No matching .npz files found.")

    if args.separate or len(paths) == 1:
        ncols = min(3, len(paths))
        nrows = math.ceil(len(paths) / ncols)
        fig, axes_grid = plt.subplots(nrows, ncols,
                                      figsize=(5 * ncols, 3.5 * nrows),
                                      squeeze=False)
        axes = [axes_grid[i // ncols][i % ncols] for i in range(len(paths))]
        for ax in axes_grid.flat[len(paths):]:
            ax.set_visible(False)
        plot_separate(paths, axes)
    else:
        fig, ax = plt.subplots(figsize=(9, 4))
        plot_overlay(paths, ax)
        ax.set_title(args.layer)

    fig.tight_layout()

    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"Saved → {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
