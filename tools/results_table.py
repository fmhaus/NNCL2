"""Print per-layer result tables for TestL_ runs (last epoch).

Usage:
    python tools/results_table.py [--saves-dir PATH] [--pattern GLOB]

Columns per layer: KNN acc@1, Classifier acc@1, Hoyer, Zero%, Ortho mean, MIG
Plus NCE loss (run-level, shown in table header).
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import pandas as pd


METRICS = [
    ("knn_acc1",   "KNN acc@1"),
    ("val_acc1",   "Clf acc@1"),
    ("hoyer",      "Hoyer"),
    ("zero_pct",   "Zero %"),
    ("ortho_mean", "Ortho mean"),
    ("mig",        "MIG"),
]

COL_W = 11  # fixed column width


def _layers(df: pd.DataFrame) -> list[str]:
    """Infer layer names from column headers, preserving backbone→proj_*→head order."""
    cols = set(df.columns)
    layers = []
    if "backbone_knn_acc1" in cols:
        layers.append("backbone")
    i = 0
    while f"proj_{i}_knn_acc1" in cols:
        layers.append(f"proj_{i}")
        i += 1
    if "head_knn_acc1" in cols:
        layers.append("head")
    return layers


def _fmt(val) -> str:
    if pd.isna(val):
        return "-".center(COL_W)
    if isinstance(val, float):
        return f"{val:.4f}".center(COL_W)
    return str(val).center(COL_W)


def print_table(name: str, df: pd.DataFrame) -> None:
    row   = df.iloc[-1]
    nce   = row.get("train_nce_loss", float("nan"))
    layers = _layers(df)

    layer_labels = {
        "backbone": "Backbone",
        "head":     "Head",
        **{f"proj_{i}": f"Proj {i}" for i in range(10)},
    }

    header_label = f"  {name}   (epoch {int(row['epoch'])})   NCE loss: {nce:.4f}"
    print(header_label)
    print("=" * (len(header_label) + 2))

    # Column header
    row_sep = "-" * (COL_W * (len(METRICS) + 1) + len(METRICS))
    label_w = max(len(layer_labels.get(l, l)) for l in layers) + 2
    header  = "Layer".ljust(label_w) + "".join(m[1].center(COL_W) for m in METRICS)
    print(header)
    print(row_sep)

    for layer in layers:
        cells = [layer_labels.get(layer, layer).ljust(label_w)]
        for col_suffix, _ in METRICS:
            col = f"{layer}_{col_suffix}"
            cells.append(_fmt(row.get(col, float("nan"))))
        print("".join(cells))

    print()


def export_csv(runs_data: list[tuple[str, pd.DataFrame]], out_path: Path) -> None:
    """Write a flat CSV: run, layer, nce_loss, then all metrics."""
    col_names = ["run", "layer", "nce_loss"] + [m[1] for m in METRICS]
    rows = []
    for name, df in runs_data:
        row    = df.iloc[-1]
        nce    = row.get("train_nce_loss", float("nan"))
        layers = _layers(df)
        layer_labels = {
            "backbone": "Backbone",
            "head":     "Head",
            **{f"proj_{i}": f"Proj {i}" for i in range(10)},
        }
        for layer in layers:
            entry = {"run": name, "layer": layer_labels.get(layer, layer), "nce_loss": nce}
            for col_suffix, label in METRICS:
                entry[label] = row.get(f"{layer}_{col_suffix}", float("nan"))
            rows.append(entry)

    # Round floats for cleaner cells
    for row in rows:
        for k, v in row.items():
            if isinstance(v, float) and not (v != v):  # skip nan
                row[k] = round(v, 4)

    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=col_names, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved -> {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--saves-dir", default="saves")
    p.add_argument("--pattern",   default="TestL*")
    p.add_argument("--csv",       default=None, metavar="PATH",
                   help="Also export results to a CSV file")
    args = p.parse_args()

    saves = Path(args.saves_dir)
    runs  = sorted(saves.glob(args.pattern),
                   key=lambda p: [int(t) if t.isdigit() else t
                                  for t in re.split(r"(\d+)", p.name)])

    if not runs:
        print(f"No runs matching '{args.pattern}' in '{saves}'")
        return

    runs_data = []
    for run_dir in runs:
        metrics_csv = run_dir / "metrics.csv"
        if not metrics_csv.exists():
            print(f"  [skip] {run_dir.name}: no metrics.csv")
            continue
        df = pd.read_csv(metrics_csv)
        runs_data.append((run_dir.name, df))
        print_table(run_dir.name, df)

    if args.csv:
        export_csv(runs_data, Path(args.csv))


if __name__ == "__main__":
    main()
