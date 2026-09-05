"""Fourier-DoA h0-conditioned MCMamba estimator.

This controlled ablation keeps the conditioning site used by the original
MCMamba-SSF baseline and changes only the DoA representation:

    one-hot + h0  ->  Fourier + h0

Here ``h0`` follows the naming used by the experiment: the direction vector
is added to the projected features immediately before LayerNorm in the first
(frequency-axis spatial) Mamba stage.  It is not a recurrent LSTM hidden
state; Mamba does not expose such a state through this implementation.

The circular features and two-layer condition MLP match Fourier-FiLM, while
the FiLM scale/bias heads are deliberately absent.  This isolates Fourier
DoA encoding from the conditioning mechanism.
"""

from __future__ import annotations

import torch
from torch import nn

from baseline_reference.mcmamba_ssf import MCMambaSSF


class _FourierH0Embedding(nn.Module):
    """Map a 180-way DoA vector to the additive first-stage condition."""

    def __init__(
        self,
        n_doa_bins: int,
        output_dim: int,
        *,
        n_harmonics: int = 4,
        condition_dim: int = 128,
    ) -> None:
        super().__init__()
        if n_harmonics < 1:
            raise ValueError("n_harmonics must be positive")
        if condition_dim < 1:
            raise ValueError("condition_dim must be positive")

        angle_degrees = torch.arange(-180, 180, 2, dtype=torch.float32)
        if angle_degrees.numel() != n_doa_bins:
            raise ValueError(
                "Fourier DoA encoding requires the 180-bin [-180, 180) "
                "grid with 2-degree resolution"
            )
        harmonics = torch.arange(1, n_harmonics + 1, dtype=torch.float32)
        phases = torch.deg2rad(angle_degrees)[:, None] * harmonics[None, :]
        basis = torch.cat((torch.sin(phases), torch.cos(phases)), dim=-1)
        self.register_buffer("doa_fourier_basis", basis)

        self.condition_encoder = nn.Sequential(
            nn.Linear(2 * n_harmonics, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim),
            nn.SiLU(),
        )
        self.output_projection = (
            nn.Identity()
            if condition_dim == output_dim
            else nn.Linear(condition_dim, output_dim)
        )

    def forward(self, target_dirs: torch.Tensor) -> torch.Tensor:
        if target_dirs.ndim != 2 or target_dirs.shape[1] != self.doa_fourier_basis.shape[0]:
            raise ValueError(
                "target_dirs must have shape "
                f"(batch, {self.doa_fourier_basis.shape[0]})"
            )
        fourier_features = target_dirs @ self.doa_fourier_basis.to(
            dtype=target_dirs.dtype
        )
        return self.output_projection(self.condition_encoder(fourier_features))


class MCMambaSSFFourierH0(MCMambaSSF):
    """Offline MCMamba with circular Fourier DoA and additive h0 conditioning."""

    def __init__(
        self,
        *args,
        doa_fourier_harmonics: int = 4,
        doa_condition_dim: int = 128,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.doa_embedding = _FourierH0Embedding(
            self.n_cond_emb_dim,
            self.freq_hidden,
            n_harmonics=doa_fourier_harmonics,
            condition_dim=doa_condition_dim,
        )

