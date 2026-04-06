"""Generate architecture diagram for the non-negative contrastive variant.

Flow:
  Main SSL chain : Image → Aug → Backbone → MLP → ReLU Transform → NT-Xent
  Classifier     : Backbone → Linear Classifier → CE loss   (detached)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches  # noqa: F401 (used in box())


def box(ax, x, y, w, h, label, sublabel=None, color="#4C72B0", fontsize=10):
    rect = mpatches.FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02", linewidth=1.2,
        edgecolor="white", facecolor=color, zorder=3,
    )
    ax.add_patch(rect)
    yo = 0.018 if sublabel else 0
    ax.text(x, y + yo, label, ha="center", va="center",
            fontsize=fontsize, color="white", fontweight="bold", zorder=4)
    if sublabel:
        ax.text(x, y - 0.04, sublabel, ha="center", va="center",
                fontsize=7.5, color="white", alpha=0.85, zorder=4)


def arrow(ax, x0, y0, x1, y1, color="#555555", dashed=False):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle="-|>", color=color, lw=1.4,
                    linestyle="dashed" if dashed else "solid",
                    mutation_scale=12,
                ), zorder=2)


fig, ax = plt.subplots(figsize=(15, 5))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
fig.patch.set_facecolor("#f5f5f5")
ax.set_facecolor("#f5f5f5")

C_IMG  = "#2d6a4f"
C_AUG  = "#1d6a96"
C_BACK = "#4C72B0"
C_FEAT = "#7b4f9e"
C_PROJ = "#c05c35"
C_LOSS = "#b5444f"
C_CLS  = "#5a7a3a"
C_EVAL = "#7a6a3a"

BW, BH = 0.10, 0.16
GAP    = 0.065
STEP   = BW + GAP

CY  = 0.62   # main chain y
CLY = 0.25   # classifier y

# Main chain: Image, Aug, Backbone, MLP, ReLU Transform, NT-Xent
x0 = 0.07
XS = [x0 + i * STEP for i in range(6)]

labels_main = [
    ("Image x",    None,          C_IMG),
    ("Aug",        "SimCLR ×2",   C_AUG),
    ("Backbone f", "ResNet18",    C_BACK),
    ("MLP",        "proj + BN",   C_PROJ),
    ("Transform",  "ReLU",        C_FEAT),
    ("NT-Xent",    "loss",        C_LOSS),
]

for x, (label, sub, color) in zip(XS[:6], labels_main):
    box(ax, x, CY, BW, BH, label, sub, color)

for i in range(len(XS) - 1):
    arrow(ax, XS[i] + BW / 2, CY, XS[i + 1] - BW / 2, CY)

# Dimension label
ax.text((XS[2] + XS[3]) / 2, CY + BH / 2 + 0.07,
        "h  (512-d)", ha="center", fontsize=7.5, color="#555555", style="italic")

# Shared weights note under backbone
ax.text(XS[2], CY - BH / 2 - 0.08,
        "shared weights\nacross views",
        ha="center", fontsize=7, color="#4477aa", style="italic")

# ── Classifier branch — branches off directly from Backbone ──────────────────
arrow(ax, XS[2], CY - BH / 2, XS[2], CLY + BH / 2, dashed=True)
ax.text(XS[2] + 0.018, (CY - BH / 2 + CLY + BH / 2) / 2,
        "detach", fontsize=7, color="#555555", style="italic")

box(ax, XS[2], CLY, BW, BH, "Classifier", "accuracy eval", color=C_CLS)
box(ax, XS[3], CLY, BW, BH, "CE loss", color=C_LOSS)
arrow(ax, XS[2] + BW / 2, CLY, XS[3] - BW / 2, CLY)

# Eval block below Transform (XS[4])
arrow(ax, XS[4], CY - BH / 2, XS[4], CLY + BH / 2, dashed=True)
box(ax, XS[4], CLY, BW, BH, "Eval",
    "sparsity · ortho\n· disentangle", color=C_EVAL)

# ── Title ────────────────────────────────────────────────────────────────────
ax.text(0.5, 0.97, "NNCL implementation",
        ha="center", va="top", fontsize=13, fontweight="bold", color="#111111")

plt.tight_layout()
plt.savefig("simclr_diagram_variant.png", dpi=180, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("Saved to simclr_diagram_variant.png")
plt.show()
