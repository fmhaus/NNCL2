"""Draw an architecture diagram of SimCLRModel.

Usage:
    python tools/model_diagram.py [--proj-layers N] [--no-projector] [--out PATH]

Requires: matplotlib
"""

import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

BOX_W     = 2.0
BOX_H     = 0.6
COL_GAP   = 0.6
Y         = 0.0
PAD       = 0.2
BRACKET_Y = Y - BOX_H / 2 - 0.16
TICK      = 0.08

COLORS = {
    "backbone": "#4C72B0",
    "proj":     "#55A868",
    "head":     "#C44E52",
    "norm":     "#8172B2",
    "loss":     "#666666",
}

FONT = dict(fontsize=8.5, ha="center", va="center", color="white", fontweight="bold")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _box(ax, cx, label, color):
    patch = FancyBboxPatch(
        (cx - BOX_W / 2, Y - BOX_H / 2), BOX_W, BOX_H,
        boxstyle="round,pad=0.05",
        facecolor=color, edgecolor="white", linewidth=1.3, zorder=3,
    )
    ax.add_patch(patch)
    ax.text(cx, Y, label, **FONT, zorder=4)


def _arrow(ax, cx_left, cx_right):
    ax.annotate(
        "", xy=(cx_right - BOX_W / 2, Y), xytext=(cx_left + BOX_W / 2, Y),
        arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.4),
        zorder=2,
    )


def _bracket(ax, cx_first, cx_last, label, color):
    x0 = cx_first - BOX_W / 2 - PAD
    x1 = cx_last  + BOX_W / 2 + PAD
    by = BRACKET_Y
    ax.plot([x0, x0, x1, x1], [by + TICK, by, by, by + TICK],
            color=color, lw=1.5, zorder=2)
    ax.text((x0 + x1) / 2, by - 0.12, label,
            ha="center", va="top", fontsize=8, color=color)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def draw(proj_layers: int, no_projector: bool, out: str):
    step = BOX_W + COL_GAP

    # Build block list: (label, color)
    blocks: list[tuple[str, str]] = []

    blocks.append(("ResNet-18\nbackbone", COLORS["backbone"]))

    proj_start = proj_end = None
    head_start = head_end = None

    if not no_projector:
        if proj_layers > 0:
            proj_start = len(blocks)
            blocks.append(("Linear\n512 \u2192 512",   COLORS["proj"]))
            blocks.append(("ReLU\n(GELU grad)", COLORS["proj"]))
            blocks.append(("L1 Norm",           COLORS["norm"]))
            proj_end = len(blocks) - 1

        head_start = len(blocks)
        blocks.append(("Linear\n512 \u2192 D",    COLORS["head"]))
        blocks.append(("ReLU\n(GELU grad)", COLORS["head"]))
        head_end = len(blocks) - 1

    blocks.append(("NT-Xent\nLoss", COLORS["loss"]))

    # Centres
    centres = [i * step for i in range(len(blocks))]

    # Figure
    total_w = centres[-1]
    fig_w   = total_w + BOX_W + 1.2
    fig_h   = 2.4
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(-BOX_W / 2 - 0.4, total_w + BOX_W / 2 + 0.4)
    ax.set_ylim(-1.1, 0.9)
    ax.axis("off")
    ax.set_aspect("equal")

    for i, (label, color) in enumerate(blocks):
        _box(ax, centres[i], label, color)

    for i in range(len(blocks) - 1):
        _arrow(ax, centres[i], centres[i + 1])

    if proj_start is not None:
        _bracket(ax, centres[proj_start], centres[proj_end],
                 f"Proj layer  \u00d7{proj_layers}", COLORS["proj"])

    if head_start is not None:
        _bracket(ax, centres[head_start], centres[head_end],
                 "Head", COLORS["head"])

    ax.set_title("SimCLR Model Architecture", fontsize=11, fontweight="bold", pad=8)

    legend_items = [
        mpatches.Patch(color=COLORS["backbone"], label="Backbone"),
        mpatches.Patch(color=COLORS["proj"],     label="Projection layer"),
        mpatches.Patch(color=COLORS["head"],     label="Head"),
        mpatches.Patch(color=COLORS["norm"],     label="Normalisation"),
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize=8,
              framealpha=0.9, edgecolor="#cccccc")

    plt.tight_layout()
    plt.savefig(out, dpi=180, bbox_inches="tight")
    print(f"Saved -> {out}")
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Draw SimCLR model architecture diagram")
    p.add_argument("--proj-layers",  default=1, type=int,
                   help="Number of hidden projection blocks (default: 1)")
    p.add_argument("--no-projector", action="store_true",
                   help="Draw backbone-only model")
    p.add_argument("--out",          default="model_diagram.png",
                   help="Output image path (default: model_diagram.png)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    draw(args.proj_layers, args.no_projector, args.out)
