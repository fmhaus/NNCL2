"""Print mean ± std over the last N epochs for a training run."""

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize the last N epochs of a run.")
    p.add_argument("--name", required=True, help="Run name — resolves to saves/<name>/metrics.csv.")
    p.add_argument("--n",    default=10,    type=int, help="Number of tail epochs to average.")
    args = p.parse_args()

    run_dir = Path("saves") / args.name
    csv     = run_dir / "metrics.csv"
    hparams = run_dir / "hparams.json"

    if not csv.exists():
        raise SystemExit(f"metrics.csv not found at '{csv}'.")

    print(f"\n{args.name}")
    print("=" * 60)

    if hparams.exists():
        import json
        params = json.loads(hparams.read_text())
        print("\nhparams:\n")
        key_w = max(len(k) for k in params)
        for k, v in params.items():
            print(f"  {k:<{key_w}}  {v}")

    df   = pd.read_csv(csv)
    tail = df.tail(args.n).drop(columns=["epoch"], errors="ignore")
    mean = tail.mean()
    std  = tail.std()

    print(f"\nmetrics  (last {min(args.n, len(df))} epochs):\n")
    col_w = max(len(c) for c in mean.index)
    for col in mean.index:
        print(f"  {col:<{col_w}}  {mean[col]:>10.4f} ± {std[col]:.4f}")
    print()


if __name__ == "__main__":
    main()
