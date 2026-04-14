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

    def __init__(self, temperature: float = 0.5, sim_clip_min: float = -1.0):
        super().__init__()
        self.temperature = temperature
        self.sim_clip_min = sim_clip_min

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        B = z1.size(0)

        # L2-normalize so dot product == cosine similarity
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # Full (2B, 2B) similarity matrix / τ
        z   = torch.cat([z1, z2], dim=0)
        sim = (z @ z.T) / self.temperature

        # Clipped similarities: There is no advantage of more than 90 degrees distance
        sim = sim.clip(min=self.sim_clip_min * self.temperature)

        # Numerator: sim(z_i, z_j) where j is the positive of i
        # z1[i] pairs with z2[i] and vice versa — stack both directions
        pos = torch.cat([(z1 * z2).sum(dim=1), (z2 * z1).sum(dim=1)]) / self.temperature  # (2B,)

        # Denominator: logsumexp over all k ≠ i  (mask out self-similarity)
        self_mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        log_denom = sim.masked_fill(self_mask, float("-inf")).logsumexp(dim=1)              # (2B,)

        # ℓ(i, j) = −log( exp(pos_i) / Σ_{k≠i} exp(sim_ik) )
        #          = −(pos_i − log_denom_i)
        return (log_denom - pos).mean()
