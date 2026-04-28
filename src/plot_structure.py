"""Visualize per-dimension and per-class structure metrics from per_class_stats/*.npz.

Usage:
    python plot_structure.py --layer encoder_out --plot R
    python plot_structure.py --names BaseMLP BaseMLPBN --layer encoder_out --plot separability
    python plot_structure.py --names BaseMLP BaseMLPBN --layer encoder_out --plot disentanglement
    python plot_structure.py --names BaseMLP BaseMLPBN --layer encoder_out --plot completeness
    python plot_structure.py --names BaseMLP BaseMLPBN --layer encoder_out --plot separability --out sep.png
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

SAVES_DIR = Path(__file__).parent.parent / "saves"


def run_label(save_dir: Path) -> str:
    name_file = save_dir / "name.txt"
    if name_file.exists():
        return name_file.read_text().strip()
    return save_dir.name


def load(name: str, layer: str) -> dict:
    p = SAVES_DIR / name / "per_class_stats" / f"{layer}_0200.npz"
    if not p.exists():
        raise FileNotFoundError(p)
    return dict(np.load(p))


# ── R heatmap ─────────────────────────────────────────────────────────────────

def plot_R(names: list[str], layer: str) -> list[plt.Figure]:
    figs = []
    for name in names:
        d = load(name, layer)
        R = d["R"]
        order = np.argsort(R.max(axis=0))[::-1]
        R_sorted = R[:, order]
        dpi = 100
        px_per_cell = 3
        C, D = R_sorted.shape
        ax_w = D * px_per_cell / dpi
        ax_h = C * px_per_cell / dpi
        fig, ax = plt.subplots(figsize=(ax_w + 2.5, ax_h + 1.5), dpi=dpi)
        im = ax.imshow(R_sorted, aspect="auto", cmap="viridis",
                       norm=mcolors.PowerNorm(gamma=0.4, vmin=0, vmax=R_sorted.max()))
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="Importance")
        ax.set_xlabel("Dimensions sorted by max importance")
        ax.set_ylabel("Class")
        ax.set_title(f"{run_label(SAVES_DIR / name)}  —  {layer}")
        fig.tight_layout()
        figs.append(fig)
    return figs


# ── Sorted per-dimension bars ─────────────────────────────────────────────────

def plot_per_dim(names: list[str], layer: str, key: str, ylabel: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 4))

    for name in names:
        d = load(name, layer)
        values = d[key]                          # (D,)
        sorted_vals = np.sort(values)[::-1]      # descending
        ax.plot(sorted_vals, label=run_label(SAVES_DIR / name), linewidth=1.2)

    ax.set_xlabel("Dimensions (sorted by score)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{key}  —  {layer}")
    ax.legend(fontsize=9)
    ax.set_xlim(0, None)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    return fig


# ── Per-class completeness bars ───────────────────────────────────────────────



# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--names", nargs="+", metavar="NAME",
                        help="Run names (default: all under saves/)")
    parser.add_argument("--layer", required=True,
                        help="Layer name, e.g. encoder_out, proj_out")
    parser.add_argument("--plot", required=True,
                        choices=["R", "separability", "disentanglement", "completeness"])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.names:
        names = args.names
    else:
        names = [p.name for p in sorted(SAVES_DIR.iterdir())
                 if p.is_dir() and (p / "per_class_stats" / f"{args.layer}_0200.npz").exists()]

    if args.plot == "R":
        figs = plot_R(names, args.layer)
        if args.out:
            for name, fig in zip(names, figs):
                out = args.out.with_stem(f"{args.out.stem}_{name}")
                fig.savefig(out, dpi=150)
                print(f"Saved → {out}")
        else:
            plt.show()
        return

    if args.plot == "separability":
        fig = plot_per_dim(names, args.layer, "separability", "Separability")
    elif args.plot == "disentanglement":
        fig = plot_per_dim(names, args.layer, "disentanglement", "Disentanglement")
    elif args.plot == "completeness":
        fig = plot_per_dim(names, args.layer, "completeness", "Completeness")

    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"Saved → {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
