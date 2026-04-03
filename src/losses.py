import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy loss (NT-Xent).

    The SimCLR contrastive loss (Chen et al., 2020 — Eq. 1).

    For a batch of B images, 2B augmented views are produced (z1, z2).
    Each view's positive pair is the other view of the same image.
    All other 2(B-1) views in the batch are negatives.

    Args:
        temperature: Scaling factor τ. Lower = sharper distribution, harder
                     negatives. Paper uses 0.5 for CIFAR, 0.07 for ImageNet.

    Input:
        z1, z2: (B, D) — L2-normalized projection head outputs. Normalization
                is applied internally so raw projector output is fine.

    Returns:
        Scalar loss averaged over all 2B views.
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        B = z1.size(0)
        device = z1.device

        # L2-normalize so dot product == cosine similarity
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # Concatenate into (2B, D) and compute full similarity matrix (2B, 2B)
        z = torch.cat([z1, z2], dim=0)
        sim = torch.mm(z, z.T) / self.temperature

        # Mask out self-similarity on the diagonal (not a valid negative)
        mask = torch.eye(2 * B, dtype=torch.bool, device=device)
        sim = sim.masked_fill(mask, float("-inf"))

        # Positive pair for row i is at i+B, and for row i+B is at i
        labels = torch.cat([torch.arange(B, 2 * B), torch.arange(B)]).to(device)

        return F.cross_entropy(sim, labels)
