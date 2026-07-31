"""Circular-DoA FiLM-conditioned MCMamba estimator for the McNet-SSF dataset.

The four processing stages follow MCMamba (Ren et al., 2025): full-band
spatial, narrow-band spatial, sub-band spectral, and full-band spectral.
The offline recipe uses Bi-Mamba in every stage. Since published MCMamba has
no target steering, this SSF adaptation encodes target DoA with circular
Fourier features and FiLM-conditions the two spatial Mamba stages.
"""

from __future__ import annotations

from typing import Sequence

import torch
from mamba_ssm import Mamba
from torch import nn
from torch.nn import functional as F


class _MambaProjection(nn.Module):
    """Input projection, Uni/Bi-Mamba, linear residual, and output projection."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        *,
        bidirectional: bool,
        activation: nn.Module,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.input_norm = nn.LayerNorm(hidden_size)
        self.forward_mamba = Mamba(
            d_model=hidden_size,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            # mamba-ssm 2.2.4's fused entry point is incompatible with the
            # installed causal-conv1d 1.6 ABI; the component CUDA kernels used
            # by the non-fused path remain accelerated and differentiable.
            use_fast_path=False,
        )
        if bidirectional:
            self.backward_mamba = Mamba(
                d_model=hidden_size,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                use_fast_path=False,
            )
            combined_size = 2 * hidden_size
        else:
            self.backward_mamba = None
            combined_size = hidden_size
        self.output_projection = nn.Linear(combined_size, output_size)
        self.linear_residual = nn.Linear(input_size, output_size)
        self.activation = activation

    def forward(
        self,
        inputs: torch.Tensor,
        film: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        projected = self.input_projection(inputs)
        projected = self.input_norm(projected)
        if film is not None:
            scale, bias = film
            if scale.shape != projected.shape or bias.shape != projected.shape:
                raise ValueError(
                    f"FiLM shapes {tuple(scale.shape)}, {tuple(bias.shape)} do "
                    f"not match projected input {tuple(projected.shape)}"
                )
            projected = projected * (1.0 + scale) + bias
        forward = self.forward_mamba(projected)
        if self.backward_mamba is not None:
            backward = torch.flip(
                self.backward_mamba(torch.flip(projected, dims=(1,))),
                dims=(1,),
            )
            encoded = torch.cat((forward, backward), dim=-1)
        else:
            encoded = forward
        outputs = self.output_projection(encoded) + self.linear_residual(inputs)
        return self.activation(outputs)


class MCMambaSSFCircularFiLM(nn.Module):
    """Offline MCMamba with circular DoA FiLM conditioning."""

    output_type = "CRM"

    def __init__(
        self,
        n_channels: int = 3,
        n_cond_emb_dim: int = 180,
        freq_hidden: int = 128,
        narrow_hidden: int = 256,
        subband_hidden: int = 384,
        fullband_hidden: int = 128,
        projection_size: int = 64,
        subband_noisy_radius: int = 3,
        subband_embedding_radius: int = 2,
        temporal_context: Sequence[int] = (5, 5),
        output_activation: str = "tanh",
        reference_channel: int = 0,
        bidirectional: bool = True,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        doa_fourier_harmonics: int = 4,
        doa_condition_dim: int = 128,
    ) -> None:
        super().__init__()
        if len(temporal_context) != 2:
            raise ValueError("temporal_context must contain (past, future)")
        if not 0 <= reference_channel < n_channels:
            raise ValueError("reference_channel is outside the microphone array")
        if output_activation != "tanh":
            raise ValueError("The SSF recipe requires tanh CRM output")
        if doa_fourier_harmonics < 1:
            raise ValueError("doa_fourier_harmonics must be positive")
        if doa_condition_dim < 1:
            raise ValueError("doa_condition_dim must be positive")

        self.n_channels = n_channels
        self.n_cond_emb_dim = n_cond_emb_dim
        self.freq_hidden = freq_hidden
        self.subband_noisy_radius = subband_noisy_radius
        self.subband_embedding_radius = subband_embedding_radius
        self.temporal_context = tuple(int(value) for value in temporal_context)
        self.reference_channel = reference_channel

        block_options = {
            "bidirectional": bidirectional,
            "d_state": mamba_d_state,
            "d_conv": mamba_d_conv,
            "expand": mamba_expand,
        }
        input_features = 2 * n_channels
        angle_degrees = torch.arange(-180, 180, 2, dtype=torch.float32)
        if angle_degrees.numel() != n_cond_emb_dim:
            raise ValueError(
                "Circular DoA basis requires the 180-bin [-180, 180) grid "
                "with 2-degree resolution"
            )
        angle_radians = torch.deg2rad(angle_degrees)
        harmonic_indices = torch.arange(
            1, doa_fourier_harmonics + 1, dtype=torch.float32
        )
        phases = angle_radians[:, None] * harmonic_indices[None, :]
        circular_basis = torch.cat((torch.sin(phases), torch.cos(phases)), dim=-1)
        self.register_buffer("doa_fourier_basis", circular_basis)
        self.doa_encoder = nn.Sequential(
            nn.Linear(2 * doa_fourier_harmonics, doa_condition_dim),
            nn.SiLU(),
            nn.Linear(doa_condition_dim, doa_condition_dim),
            nn.SiLU(),
        )
        self.freq_film = nn.Linear(doa_condition_dim, 2 * freq_hidden)
        self.narrow_film = nn.Linear(doa_condition_dim, 2 * narrow_hidden)
        # Start from an unconditional normalized representation and let
        # steering strength grow smoothly during optimization.
        nn.init.zeros_(self.freq_film.weight)
        nn.init.zeros_(self.freq_film.bias)
        nn.init.zeros_(self.narrow_film.weight)
        nn.init.zeros_(self.narrow_film.bias)
        self.freq = _MambaProjection(
            input_features,
            freq_hidden,
            projection_size,
            activation=nn.ReLU(),
            **block_options,
        )
        self.narrow = _MambaProjection(
            projection_size + input_features,
            narrow_hidden,
            projection_size,
            activation=nn.ReLU(),
            **block_options,
        )
        subband_input = (
            (2 * subband_embedding_radius + 1) * projection_size
            + 2 * subband_noisy_radius
            + 1
        )
        self.subband = _MambaProjection(
            subband_input,
            subband_hidden,
            projection_size,
            activation=nn.ReLU(),
            **block_options,
        )
        fullband_input = projection_size + sum(self.temporal_context) + 1
        self.fullband = _MambaProjection(
            fullband_input,
            fullband_hidden,
            2,
            activation=nn.Tanh(),
            **block_options,
        )

    @staticmethod
    def _frequency_context(values: torch.Tensor, radius: int) -> torch.Tensor:
        if radius == 0:
            return values
        batch, frames, frequencies, features = values.shape
        packed = values.permute(0, 1, 3, 2).reshape(
            batch * frames, features, frequencies, 1
        )
        packed = torch.cat(
            (packed[:, :, :radius], packed, packed[:, :, -radius:]), dim=2
        )
        packed = F.unfold(packed, kernel_size=(2 * radius + 1, 1))
        return packed.reshape(batch, frames, -1, frequencies).permute(0, 1, 3, 2)

    def _temporal_magnitude_context(self, magnitude: torch.Tensor) -> torch.Tensor:
        past, future = self.temporal_context
        packed = magnitude.permute(0, 2, 3, 1)
        packed = F.pad(packed, (past, future))
        packed = packed.reshape(magnitude.shape[0] * magnitude.shape[2], 1, -1, 1)
        packed = F.unfold(packed, kernel_size=(past + future + 1, 1))
        return packed.reshape(
            magnitude.shape[0], magnitude.shape[2], -1, magnitude.shape[1]
        ).permute(0, 3, 1, 2)

    def forward(
        self,
        inputs: torch.Tensor,
        target_dirs: torch.Tensor,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        del device
        batch, channels, frequencies, frames = inputs.shape
        expected_channels = 2 * self.n_channels
        if channels != expected_channels:
            raise ValueError(f"Expected {expected_channels} real/imag channels, got {channels}")
        if target_dirs.shape != (batch, self.n_cond_emb_dim):
            raise ValueError(
                f"Expected DoA encoding {(batch, self.n_cond_emb_dim)}, "
                f"got {tuple(target_dirs.shape)}"
            )

        noisy = inputs.permute(0, 3, 2, 1)
        reference_real = noisy[..., self.reference_channel]
        reference_imag = noisy[..., self.n_channels + self.reference_channel]
        reference_magnitude = torch.sqrt(
            reference_real.square() + reference_imag.square() + 1e-12
        ).unsqueeze(-1)

        freq_input = noisy.reshape(batch * frames, frequencies, channels)
        circular_doa = target_dirs @ self.doa_fourier_basis.to(target_dirs.dtype)
        doa_condition = self.doa_encoder(circular_doa)

        freq_scale, freq_bias = self.freq_film(doa_condition).chunk(2, dim=-1)
        freq_scale = freq_scale[:, None, None, :].expand(
            batch, frames, frequencies, self.freq_hidden
        )
        freq_bias = freq_bias[:, None, None, :].expand(
            batch, frames, frequencies, self.freq_hidden
        )
        features = self.freq(
            freq_input,
            (
                freq_scale.reshape(
                    batch * frames, frequencies, self.freq_hidden
                ),
                freq_bias.reshape(
                    batch * frames, frequencies, self.freq_hidden
                ),
            ),
        ).reshape(batch, frames, frequencies, -1)

        narrow_input = torch.cat((features, noisy), dim=-1)
        narrow_input = narrow_input.permute(0, 2, 1, 3).reshape(
            batch * frequencies, frames, -1
        )
        narrow_scale, narrow_bias = self.narrow_film(doa_condition).chunk(2, dim=-1)
        narrow_scale = narrow_scale[:, None, None, :].expand(
            batch, frequencies, frames, self.narrow.hidden_size
        )
        narrow_bias = narrow_bias[:, None, None, :].expand(
            batch, frequencies, frames, self.narrow.hidden_size
        )
        features = self.narrow(
            narrow_input,
            (
                narrow_scale.reshape(
                    batch * frequencies, frames, self.narrow.hidden_size
                ),
                narrow_bias.reshape(
                    batch * frequencies, frames, self.narrow.hidden_size
                ),
            ),
        ).reshape(
            batch, frequencies, frames, -1
        ).permute(0, 2, 1, 3)

        embedding_context = self._frequency_context(
            features, self.subband_embedding_radius
        )
        magnitude_context = self._frequency_context(
            reference_magnitude, self.subband_noisy_radius
        )
        subband_input = torch.cat((embedding_context, magnitude_context), dim=-1)
        subband_input = subband_input.permute(0, 2, 1, 3).reshape(
            batch * frequencies, frames, -1
        )
        features = self.subband(subband_input).reshape(
            batch, frequencies, frames, -1
        ).permute(0, 2, 1, 3)

        magnitude_context = self._temporal_magnitude_context(reference_magnitude)
        fullband_input = torch.cat((features, magnitude_context), dim=-1)
        fullband_input = fullband_input.reshape(batch * frames, frequencies, -1)
        mask = self.fullband(fullband_input).reshape(batch, frames, frequencies, 2)
        return mask.permute(0, 3, 2, 1).contiguous()
