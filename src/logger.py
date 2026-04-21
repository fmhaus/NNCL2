"""Training logger: hparams, metrics CSV, checkpoints, console output, openbayestool."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    from openbayestool import log_metric, clear_metric # type: ignore
    _OPENBAYESTOOL_AVAILABLE = True
except ImportError:
    def log_metric(*args, **kwargs): pass  # type: ignore[misc]
    def clear_metric(*args, **kwargs): pass  # type: ignore[misc]
    _OPENBAYESTOOL_AVAILABLE = False


# Metric keys logged every epoch — cleared at init to avoid stale data from prior runs.
_METRIC_KEYS = [
    "train_nce_loss",
    "train_class_loss",
    "train_acc1_epoch",
    "train_acc5_epoch",
    "grad_backbone_norm",
    "grad_feat_norm",
    "grad_proj_out_norm",
    "val_knn_acc1",
    "val_knn_acc5",
    "val_loss",
    "val_acc1",
    "val_acc5",
    "feat_l1",
    "feat_l2",
    "feat_hoyer",
    "lr",
    "epoch_time_s",
]


class TrainingLogger:
    """Manages all training output for a single run.

    Creates under save_dir/:
      hparams.json    — argparse args, written once at init
      metrics.csv     — one row per epoch, appended and saved each epoch
      state_XXXX.ckpt — latest checkpoint only (previous deleted on save)

    Optionally integrates with openbayestool:
      - metrics logged each epoch with log_metric
      - all metric keys cleared at init with clear_metric
    """

    def __init__(self, save_dir: Path, args, console_log: bool = True):
        self.save_dir          = save_dir
        self.console_log       = console_log
        self.use_openbayestool = _OPENBAYESTOOL_AVAILABLE
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self._rows: list[dict] = []
        self._last_ckpt: Path | None = None

        (save_dir / "hparams.json").write_text(json.dumps(vars(args), indent=2))

        # Restore existing metrics on resume so new epochs append correctly
        metrics_path = save_dir / "metrics.csv"
        if metrics_path.exists():
            self._rows = pd.read_csv(metrics_path).to_dict(orient="records")

        print(f"Run directory: {save_dir}")

        if self.use_openbayestool:
            # Clear stale metric data from any previous run
            for key in _METRIC_KEYS:
                clear_metric(key)

    def log(self, epoch: int, metrics: dict[str, float]) -> None:
        """Log metrics for one epoch — console + metrics.csv + openbayestool."""
        row  = {"epoch": epoch, **metrics}
        self._rows.append(row)

        pd.DataFrame(self._rows).to_csv(self.save_dir / "metrics.csv", index=False)

        if self.console_log:
            parts = [f"Epoch {epoch:>3}"] + [
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in metrics.items()
            ]
            print(" | ".join(parts))

        if self.use_openbayestool:
            for key, value in metrics.items():
                log_metric(key, value)

    def log_activation_histogram(self, epoch: int, name: str, tensor: torch.Tensor, bins: int = 100) -> None:
        """Save a histogram of tensor values to histograms/<name>_<epoch>.npz."""
        values = tensor.detach().float().cpu().numpy().ravel()
        counts, edges = np.histogram(values, bins=bins)
        hist_dir = self.save_dir / "histograms"
        hist_dir.mkdir(exist_ok=True)
        np.savez_compressed(hist_dir / f"{name}_{epoch:04d}.npz", counts=counts, edges=edges)

    def save_checkpoint(self, epoch: int, state: dict) -> None:
        """Save state dict, then delete the previous checkpoint."""
        path = self.save_dir / f"state_{epoch:04d}.ckpt"
        torch.save(state, path)
        if self._last_ckpt is not None and self._last_ckpt.exists():
            self._last_ckpt.unlink()
        self._last_ckpt = path
