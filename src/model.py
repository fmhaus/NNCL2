from typing import Literal
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
        feature_transform: None | Literal["relu", "simplex_proj", "softmax", "L1_norm"] = None,
        use_projector: bool = True
    ):
        super().__init__()

        base = resnet18(weights=None)

        if image_size <= 64:
            # Replace 7×7 stride-2 conv + maxpool with 3×3 stride-1 conv.
            # Preserves spatial resolution for small images (CIFAR-100: 32, TinyImageNet: 64).
            base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            base.maxpool = nn.Identity() # type: ignore

        self.backbone = nn.Sequential(*list(base.children())[:-1])  # (B, 512, 1, 1)

        if feature_transform is None:
            self.feature_transform = nn.Identity()
        elif feature_transform == "relu":
            self.feature_transform = ReluGeluGrad()
        elif feature_transform == "softmax":
            self.feature_transform = nn.Softmax(dim=-1)
        elif feature_transform == "L1_norm":
            self.feature_transform = ShiftL1Norm()
            


        if use_projector:
            self.projector = nn.Sequential(
                nn.Linear(BACKBONE_DIM, proj_hidden),
                nn.BatchNorm1d(proj_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(proj_hidden, proj_dim),
                nn.BatchNorm1d(proj_dim),
            )
        else:
            self.projector = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns projected embeddings — used for the SSL contrastive loss."""
        return self.projector(self.encode(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Returns backbone embeddings (512-dim) — used for downstream evaluation."""
        return self.feature_transform(self.backbone(x).flatten(1))


class ShiftL1Norm(nn.Module):
    """Shift to non-negative then L1-normalize: (x - min) / sum(x - min)."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x - x.min(dim=1, keepdim=True).values
        return x / x.sum(dim=1, keepdim=True).clamp(min=1e-8)


class ReluGeluGrad(nn.Module):
    """Forward pass of ReLU, backward pass (gradient) of GELU.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x).detach() + F.gelu(x) - F.gelu(x).detach()
    


class LinearClassifier(nn.Module):
    """Linear head for linear probing or fine-tuning on top of a frozen SimCLR backbone.

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

    def __init__(self, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(BACKBONE_DIM, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)
