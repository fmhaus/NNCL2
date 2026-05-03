"""Compute SEPIN disentanglement metrics for a single layer of a legacy checkpoint.

Usage:
  uv run python main_disent.py --path trained_models/simclr/hywyrz38 --layer proj
  uv run python main_disent.py --path trained_models/simclr/ImageNet100-CL --layer proj --data-root /data/imagenet100/train
"""

import argparse
import json
from pathlib import Path

import torch

from main_eval2 import (
    _DATASET_STATS,
    _build_aug_transform,
    _make_loader,
    compute_disentanglement,
    find_checkpoint,
    load_legacy_checkpoint,
)

LAYERS = ("encoder", "proj_hidden", "proj")
HIDDEN_SLICE = 2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--path",   required=True,
                   help="Run directory, e.g. trained_models/simclr/hywyrz38")
    p.add_argument("--layer",  required=True, choices=LAYERS)
    p.add_argument("--epoch",  default=None, type=int,
                   help="Checkpoint epoch. Default: highest available.")
    p.add_argument("--data-root",      default=None,
                   help="Path to training data root. Required when args.json has a relative path.")
    p.add_argument("--num-workers",    default=None, type=int)
    p.add_argument("--batch-size",     default=None, type=int)
    p.add_argument("--nce-batch-size", default=256,  type=int,
                   help="Mini-batch size for NCE computation.")
    p.add_argument("--full-nce", action="store_true",
                   help="Use full-dataset NCE (all 2N negatives per example).")
    p.add_argument("--full-nce-chunk", default=512, type=int,
                   help="Row-chunk size for full-NCE memory management.")
    p.add_argument("--prenorm-loo", action="store_true",
                   help="Normalize the full D-dim vector before slicing. "
                        "LOO vectors have norm sqrt(1-z_i^2) — feature i's energy "
                        "is absent without compensating re-normalization.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    run_dir   = Path(args.path)
    ckpt_path = find_checkpoint(run_dir, args.epoch)
    run_args  = json.loads((run_dir / "args.json").read_text())

    data_root   = args.data_root or run_args["data"]["train_path"]
    num_workers = args.num_workers if args.num_workers is not None else run_args["data"]["num_workers"]
    batch_size  = args.batch_size or run_args["optimizer"]["batch_size"]
    temperature = run_args["method_kwargs"].get("temperature", 0.2)
    non_neg     = run_args["method_kwargs"].get("non_neg")
    if non_neg == "None":
        non_neg = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"ckpt={ckpt_path.name}  device={device}  layer={args.layer}  "
          f"temperature={temperature}  non_neg={non_neg}")

    dataset = run_args["data"]["dataset"]
    if dataset not in _DATASET_STATS:
        raise SystemExit(f"Unsupported dataset '{dataset}'. Supported: {list(_DATASET_STATS)}")
    mean, std = _DATASET_STATS[dataset]

    aug_t = _build_aug_transform(run_args["augmentations"][0], mean, std)
    # val_path unused for two_view train loader; pass data_root as placeholder
    train_loader_tv = _make_loader(
        dataset, data_root, data_root, train=True,
        batch_size=batch_size, num_workers=num_workers, transform=aug_t, two_view=True,
    )

    model, epoch = load_legacy_checkpoint(ckpt_path, run_args)
    model.to(device).eval()
    print(f"epoch={epoch}")

    if args.layer == "encoder":
        encode_fn = lambda imgs: model.encode(imgs)
    elif args.layer == "proj_hidden":
        encode_fn = lambda imgs, s=HIDDEN_SLICE: model.projector[:s](model.encode(imgs))
    else:
        encode_fn = lambda imgs: model.projector(model.encode(imgs))

    results = compute_disentanglement(
        encode_fn, train_loader_tv, device,
        temperature=temperature, nce_batch_size=args.nce_batch_size,
        use_full_nce=args.full_nce, full_nce_chunk=args.full_nce_chunk,
        prenorm_loo=args.prenorm_loo,
    )

    print(f"\n[{args.layer}]")
    for k, v in results.items():
        print(f"  {k:<12} {v:.4f}")


if __name__ == "__main__":
    main()
