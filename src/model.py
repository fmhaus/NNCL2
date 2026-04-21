import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


BACKBONE_DIM = 512  # ResNet18 output dimension


class SimCLRModel(nn.Module):
    """ResNet18 encoder with a SimCLR projection head.

    Architecture (Chen et al., 2020):
        backbone  f : image → (B, 512)          — used for downstream tasks
        projector g : (B, 512) → (B, proj_dim)  — used only during SSL training

    The projector is a 2-layer BN-MLP as specified in the paper.
    After pretraining, attach a linear classifier to the backbone output,
    not the projector output (the paper ablates this — projector hurts linear probe).

    Args:
        proj_hidden: Hidden dim of the projection MLP (paper uses 2048 for ResNet50;
                     512 is standard for ResNet18).
        proj_dim:    Output dim of the projector (paper: 128).
    """

    def __init__(
        self,
        proj_hidden: int = 512,
        proj_dim: int = 128,
        image_size: int = 224,
        projector: str = "mlp",   # "none" | "mlp" | "mlp-bn"
    ):
        super().__init__()

        self.backbone = resnet18(weights=None)

        if image_size <= 64:
            # Replace 7×7 stride-2 conv + maxpool with 3×3 stride-1 conv.
            # Preserves spatial resolution for small images (CIFAR-100: 32, TinyImageNet: 64).
            self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.backbone.maxpool = nn.Identity() # type: ignore

        # remove fast forward projector
        self.backbone.fc = nn.Identity() # type: ignore

        if projector == "none":
            self.projector: nn.Module = nn.Identity()
        elif projector == "mlp":
            self.projector = nn.Sequential(
                nn.Linear(BACKBONE_DIM, proj_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(proj_hidden, proj_dim),
            )
        elif projector == "mlp-bn":
            # SimCLR paper: BN after each linear layer
            self.projector = nn.Sequential(
                nn.Linear(BACKBONE_DIM, proj_hidden, bias=False),
                nn.BatchNorm1d(proj_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(proj_hidden, proj_dim, bias=False),
                nn.BatchNorm1d(proj_dim),
            )
        else:
            raise ValueError(f"Unknown projector '{projector}'. Choose: none, mlp, mlp-bn.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns projected embeddings — used for the SSL contrastive loss."""
        return self.projector(self.encode(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Returns backbone embeddings after feature_transform. Shape: (B, 512)."""
        return self.backbone(x).flatten(1)


class LinearClassifier(nn.Module):
    """
    Linear head for linear probing or fine-tuning on top of a frozen SimCLR backbone.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(BACKBONE_DIM, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)
