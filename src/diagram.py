"""Generate a SimCLR architecture diagram."""

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

BW, BH = 0.10, 0.16
GAP    = 0.065          # gap between box edges
STEP   = BW + GAP       # center-to-center distance

CY  = 0.62             # main chain y
CLY = 0.25             # classifier y

# Main chain x-centers
x0 = 0.07
XS = [x0 + i * STEP for i in range(6)]
# XS: Image, Aug, Backbone, Transform, Projector, NT-Xent

labels_main = [
    ("Image x",    None,               C_IMG),
    ("Aug",        "SimCLR ×2",        C_AUG),
    ("Backbone f", "ResNet18",         C_BACK),
    ("Transform",  "ReLU / L1…",       C_FEAT),
    ("Projector g","MLP + BN",         C_PROJ),
    ("NT-Xent",    "loss",             C_LOSS),
]

for x, (label, sub, color) in zip(XS, labels_main):
    box(ax, x, CY, BW, BH, label, sub, color)

# Arrows between main chain boxes
for i in range(len(XS) - 1):
    arrow(ax, XS[i] + BW / 2, CY, XS[i + 1] - BW / 2, CY)

# Dimension labels above arrows
ax.text((XS[3] + XS[4]) / 2, CY + BH / 2 + 0.07,
        "h  (512-d)", ha="center", fontsize=7.5, color="#555555", style="italic")
ax.text((XS[4] + XS[5]) / 2, CY + BH / 2 + 0.07,
        "z  (128-d)", ha="center", fontsize=7.5, color="#555555", style="italic")

# "shared weights" note under backbone
ax.text(XS[2], CY - BH / 2 - 0.08,
        "shared weights\nacross views",
        ha="center", fontsize=7, color="#4477aa", style="italic")

# ── Online classifier branch ─────────────────────────────────────────────────
# Dashed arrow down from Transform
arrow(ax, XS[3], CY - BH / 2, XS[3], CLY + BH / 2, dashed=True)
ax.text(XS[3] + 0.018, (CY - BH / 2 + CLY + BH / 2) / 2,
        "detach", fontsize=7, color="#555555", style="italic")

# Classifier box aligned under Transform
box(ax, XS[3], CLY, BW, BH, "Classifier", "accuracy eval", color=C_CLS)

# CE loss box aligned under Projector
box(ax, XS[4], CLY, BW, BH, "CE loss", color=C_LOSS)

# Arrow from Classifier to CE loss
arrow(ax, XS[3] + BW / 2, CLY, XS[4] - BW / 2, CLY)

# ── Title ────────────────────────────────────────────────────────────────────
ax.text(0.5, 0.97, "SimCLR — SSL Pre-training Architecture",
        ha="center", va="top", fontsize=13, fontweight="bold", color="#111111")


plt.tight_layout()
plt.savefig("simclr_diagram.png", dpi=180, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("Saved to simclr_diagram.png")
plt.show()
