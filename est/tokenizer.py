import torch
from torch import nn
from typing import Tuple


def cartesian_to_spherical(points: torch.Tensor) -> torch.Tensor:
    """Convert xyz to (r, theta, phi). theta: inclination, phi: azimuth.

    Args:
        points: [N, 3]
    Returns:
        spherical coordinates tensor [N, 3]
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = torch.sqrt(x ** 2 + y ** 2 + z ** 2 + 1e-9)
    theta = torch.acos(torch.clamp(z / r, -1.0, 1.0))
    phi = torch.atan2(y, x)
    return torch.stack([r, theta, phi], dim=-1)


class SphericalWindowTokenizer(nn.Module):
    """Tokenizes a point cloud into spherical windows and aggregates features."""

    def __init__(
        self,
        theta_bins: int = 24,
        phi_bins: int = 48,
        radial_bins: int = 4,
        feature_dim: int = 4,
        embed_dim: int = 64,
    ) -> None:
        super().__init__()
        self.theta_bins = theta_bins
        self.phi_bins = phi_bins
        self.radial_bins = radial_bins
        self.linear = nn.Linear(feature_dim, embed_dim)

    def forward(self, points: torch.Tensor, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize and pool into windows.

        Args:
            points: [N, 3] xyz positions.
            features: [N, F] features (e.g., intensity, ring).
        Returns:
            tokens: [W, C] where W is number of windows.
            window_indices: indices of points assigned to each window.
        """
        spherical = cartesian_to_spherical(points)
        r, theta, phi = spherical[:, 0], spherical[:, 1], spherical[:, 2]

        theta_bin = torch.clamp((theta / torch.pi * self.theta_bins).long(), 0, self.theta_bins - 1)
        phi_normalized = (phi + torch.pi) / (2 * torch.pi)
        phi_bin = torch.clamp((phi_normalized * self.phi_bins).long(), 0, self.phi_bins - 1)
        radial_norm = torch.clamp(r / (r.max() + 1e-6), 0, 1)
        radial_bin = torch.clamp((radial_norm * self.radial_bins).long(), 0, self.radial_bins - 1)

        window_id = theta_bin * (self.phi_bins * self.radial_bins) + phi_bin * self.radial_bins + radial_bin
        num_windows = self.theta_bins * self.phi_bins * self.radial_bins
        tokens = torch.zeros(num_windows, features.size(1), device=features.device)
        counts = torch.zeros(num_windows, device=features.device)

        tokens.index_add_(0, window_id, features)
        counts.index_add_(0, window_id, torch.ones_like(radial_bin, dtype=counts.dtype))
        counts = torch.clamp(counts, min=1.0)
        tokens = tokens / counts.unsqueeze(1)
        embedded = self.linear(tokens)
        return embedded, window_id
