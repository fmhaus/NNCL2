from typing import cast
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


BACKBONE_DIM = 512  # ResNet18 output dimension


# ---------------------------------------------------------------------------
# Atomic norm modules (used inside nn.Sequential)
# ---------------------------------------------------------------------------

class _ReluGeluGrad(nn.Module):
    """ReLU forward, GELU gradient — avoids hard zeros in backprop."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x).detach() + F.gelu(x) - F.gelu(x).detach()

class _L1Norm(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / x.sum(dim=-1, keepdim=True).clamp(min=1e-8)

class _L2Norm(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=-1)


# ---------------------------------------------------------------------------
# Projection head
# ---------------------------------------------------------------------------

class SimplexProjector(nn.Module):
    """Variable-depth projection head with constant hidden width.

    Shape: in_dim → in_dim → ... → in_dim → out_dim
           |-------- num_layers blocks --------|head|

    Hidden layers : Linear(x→x) → ReLU (GELU grad) → L1Norm  (simplex)
    Final layer   : Linear(x→c) → ReLU (GELU grad) → L2Norm  (non-negative unit sphere)

    Args:
        in_dim:     Input (and hidden) dimension.
        out_dim:    Output dimension of the head.
        num_layers: Number of hidden blocks before the head. 0 = head only.
    """

    def __init__(self, in_dim: int, out_dim: int, num_layers: int = 1):
        super().__init__()

        self.projector = nn.ModuleList(
            nn.Sequential(nn.Linear(in_dim, in_dim), _ReluGeluGrad(), _L1Norm())
            for _ in range(num_layers)
        )
        self.head = nn.Sequential(nn.Linear(in_dim, out_dim), _ReluGeluGrad(), _L2Norm())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.projector:
            x = block(x)
        return self.head(x)

    def forward_intermediates(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Output after each block, including the head. Length = len(dims) - 1."""
        outputs = []
        for block in self.projector:
            x = block(x)
            outputs.append(x)
        outputs.append(self.head(x))
        return outputs


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class SimCLRModel(nn.Module):
    """ResNet18 backbone with a simplex projection head.

    backbone  f : image → (B, 512)        — used for downstream tasks
    projector g : (B, 512) → (B, out_dim) — used only during SSL training

    Args:
        proj_out_dim:  Output dimension of the projection head.
        proj_layers:   Number of hidden x→x blocks before the final x→c head.
                       0 = head only (single linear projection).
                       Set to None to disable the projector entirely.
    """

    def __init__(
        self,
        proj_out_dim: int = 128,
        proj_layers: int | None = 1,
        image_size: int = 224,
    ):
        super().__init__()

        base = resnet18(weights=None)

        if image_size <= 64:
            # Replace 7×7 stride-2 conv + maxpool with 3×3 stride-1 conv.
            # Preserves spatial resolution for small images (CIFAR-100: 32, TinyImageNet: 64).
            base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            base.maxpool = nn.Identity() # type: ignore

        base.fc = nn.Identity()  # type: ignore
        self.backbone = base

        if proj_layers is None:
            self.projector = nn.Identity()
        else:
            self.projector = SimplexProjector(BACKBONE_DIM, proj_out_dim, proj_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns projected embeddings — used for the SSL contrastive loss."""
        return self.projector(self.encode(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Returns backbone embeddings. Shape: (B, 512)."""
        return self.backbone(x)

    def encode_all(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Returns features at every level: [backbone, proj_0, ..., head]."""
        feat = self.encode(x)
        if isinstance(self.projector, SimplexProjector):
            return [feat] + self.projector.forward_intermediates(feat)
        return [feat]

    @property
    def feature_names(self) -> list[str]:
        if not isinstance(self.projector, SimplexProjector):
            return ["backbone"]
        n_hidden = len(self.projector.projector)
        return ["backbone"] + [f"proj_{i}" for i in range(n_hidden)] + ["head"]

    @property
    def feature_dims(self) -> list[int]:
        if not isinstance(self.projector, SimplexProjector):
            return [BACKBONE_DIM]
        n_hidden = len(self.projector.projector)
        out_dim = cast(nn.Linear, self.projector.head[0]).out_features
        return [BACKBONE_DIM] * (n_hidden + 1) + [out_dim]


# ---------------------------------------------------------------------------
# Linear classifier (linear probe / fine-tuning)
# ---------------------------------------------------------------------------

class LinearClassifier(nn.Module):
    """Linear head for linear probing or fine-tuning on top of a frozen backbone.

    Usage:
        model = SimCLRModel()
        # ... load pretrained weights ...
        classifier = LinearClassifier(num_classes=100)

        model.backbone.eval()
        for param in model.backbone.parameters():
            param.requires_grad = False

        features = model.encode(images)   # (B, 512), no grad through backbone
        logits   = classifier(features)   # (B, num_classes)
    """

    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)
