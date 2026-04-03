import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

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
        pretrained:  Load ImageNet weights for the backbone.
        proj_hidden: Hidden dim of the projection MLP (paper uses 2048 for ResNet50;
                     512 is standard for ResNet18).
        proj_dim:    Output dim of the projector (paper: 128).
    """

    def __init__(
        self,
        pretrained: bool = True,
        proj_hidden: int = 512,
        proj_dim: int = 128,
    ):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        base = resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # (B, 512, 1, 1)

        self.projector = nn.Sequential(
            nn.Linear(BACKBONE_DIM, proj_hidden),
            nn.BatchNorm1d(proj_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden, proj_dim),
            nn.BatchNorm1d(proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns projected embeddings — used for the SSL contrastive loss."""
        return self.projector(self.encode(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Returns backbone embeddings (512-dim) — used for downstream evaluation."""
        return self.backbone(x).flatten(1)


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
