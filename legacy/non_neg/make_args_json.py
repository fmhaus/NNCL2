"""Convert a pretrain launch YAML into an args.json compatible with our eval scripts.

Resolves Hydra defaults (augmentations, wandb sub-yamls) and runs parse_cfg to fill
in the same defaults that a real training run would produce.

Usage:
  uv run python make_args_json.py scripts/pretrain/cifar/simclr.yaml --output trained_models/simclr/myrun
  uv run python make_args_json.py scripts/pretrain/cifar/simclr.yaml --output myrun method_kwargs.non_neg=rep_relu
"""

import argparse
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf, DictConfig

from solo.args.pretrain import parse_cfg


def _resolve_defaults(cfg: DictConfig, config_dir: Path) -> DictConfig:
    """Load and merge sub-yamls listed in the Hydra defaults block."""
    defaults = OmegaConf.to_container(cfg.get("defaults", []), resolve=False)

    for entry in defaults:
        if not isinstance(entry, dict):
            continue  # "_self_" and override strings, skip
        for key, filename in entry.items():
            if key.startswith("override "):
                continue
            sub_yaml = config_dir / key / filename
            if not sub_yaml.exists():
                print(f"  [warn] defaults sub-yaml not found, skipping: {sub_yaml}")
                continue
            sub_cfg = OmegaConf.load(sub_yaml)
            cfg = OmegaConf.merge(cfg, OmegaConf.create({key: sub_cfg}))

    return cfg


def _apply_overrides(cfg: DictConfig, overrides: list) -> DictConfig:
    """Apply Hydra-style dotlist overrides (e.g. method_kwargs.non_neg=rep_relu)."""
    for override in overrides:
        if "=" not in override:
            print(f"  [warn] skipping malformed override (expected key=value): {override}")
            continue
        key, value = override.split("=", 1)
        # Parse value as YAML scalar so booleans/ints/null are typed correctly
        parsed = OmegaConf.create({key.split(".")[-1]: OmegaConf.create(f"x: {value}")["x"]})
        # Rebuild nested path
        parts = key.split(".")
        nested = parsed
        for part in reversed(parts[:-1]):
            nested = OmegaConf.create({part: nested})
        cfg = OmegaConf.merge(cfg, nested)
    return cfg


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("config", help="Path to pretrain launch yaml.")
    p.add_argument("--output", required=True,
                   help="Directory to write args.json into.")
    p.add_argument("overrides", nargs="*",
                   help="Hydra-style overrides, e.g. method_kwargs.non_neg=rep_relu")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")

    cfg = OmegaConf.load(config_path)
    OmegaConf.set_struct(cfg, False)

    # Resolve defaults (augmentations, wandb, …)
    cfg = _resolve_defaults(cfg, config_path.parent)

    # Drop Hydra-specific keys before parse_cfg
    for key in ("defaults", "hydra"):
        if key in cfg:
            del cfg[key]

    # Apply CLI overrides
    if args.overrides:
        cfg = _apply_overrides(cfg, args.overrides)

    # Fill in all defaults exactly as the real training run does
    cfg = parse_cfg(cfg)

    # Serialise — OmegaConf.to_container converts to plain Python dicts/lists
    out_dict = OmegaConf.to_container(cfg, resolve=True)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "args.json"
    out_path.write_text(json.dumps(out_dict, indent=2, default=str))
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
