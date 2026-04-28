"""Aggregate eval_metrics.csv from multiple named runs into a single Excel file.

Usage:
    python aggregate_eval.py                          # all runs under saves/
    python aggregate_eval.py --names BaseMLP BaseMLPBN
    python aggregate_eval.py --out results/combined.xlsx
"""

import argparse
import pathlib
import pandas as pd

SAVES_DIR = pathlib.Path(__file__).parent.parent / "saves"


def collect(names: list[str] | None, saves_dir: pathlib.Path) -> pd.DataFrame:
    if names:
        paths = [saves_dir / n / "eval_metrics.csv" for n in names]
    else:
        paths = sorted(saves_dir.glob("*/eval_metrics.csv"))

    frames = []
    for p in paths:
        if not p.exists():
            print(f"WARNING: {p} not found, skipping")
            continue
        df = pd.read_csv(p)
        df.insert(0, "name", p.parent.name)
        frames.append(df)

    if not frames:
        raise FileNotFoundError("No eval_metrics.csv files found.")

    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--names", nargs="+", metavar="NAME",
                        help="Run names to include (default: all under saves/)")
    parser.add_argument("--saves-dir", type=pathlib.Path, default=SAVES_DIR)
    parser.add_argument("--out", type=pathlib.Path,
                        default=SAVES_DIR.parent / "eval_combined.xlsx")
    args = parser.parse_args()

    combined = collect(args.names, args.saves_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_excel(args.out, index=False)
    print(f"Wrote {len(combined)} rows ({combined['name'].nunique()} runs) → {args.out}")


if __name__ == "__main__":
    main()
