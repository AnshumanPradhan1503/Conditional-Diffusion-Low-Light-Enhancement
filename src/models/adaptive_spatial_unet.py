
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.unet import SinusoidalTimeEmbedding, ResidualBlock


class AdaptiveConditionGate(nn.Module):
    """
    Produces a channel-wise adaptive gate for condition features.

    The gate is predicted from the condition feature itself and constrained
    to the range [0, 1] using a sigmoid activation.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()

        hidden_channels = max(channels // 4, 8)

        self.gate = nn.Sequential(
            nn.Conv2d(
                channels,
                hidden_channels,
                kernel_size=1,
            ),
            nn.SiLU(),
            nn.Conv2d(
                hidden_channels,
                channels,
                kernel_size=1,
            ),
            nn.Sigmoid(),
        )

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        return self.gate(condition)


class AdaptiveConditionBlock(nn.Module):
    """
    Extracts spatial condition features and applies channel-wise
    input-adaptive gating.
    """

    def __init__(
        self,
        condition_channels: int,
        feature_channels: int,
    ) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(
                condition_channels,
                feature_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.GroupNorm(8, feature_channels),
            nn.SiLU(),
            nn.Conv2d(
                feature_channels,
                feature_channels,
                kernel_size=3,
                padding=1,
            ),
        )

        self.adaptive_gate = AdaptiveConditionGate(feature_channels)

    def forward(
        self,
        condition: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(condition)

        gate = self.adaptive_gate(features)

        gated_condition = features * gate

        return gated_condition, gate


class AdaptiveSpatialConditionalUNet(nn.Module):
    """
    Conditional DDPM U-Net with channel-wise, input-adaptive
    multi-scale spatial conditioning.

    Conditioning is injected at:
        - 256x256
        - 128x128
        - 64x64

    The adaptive gate is predicted independently at each scale.
    """

    def __init__(
        self,
        image_channels: int = 3,
        condition_channels: int = 3,
        base_channels: int = 32,
        time_dim: int = 128,
    ) -> None:
        super().__init__()

        self.time_embedding = SinusoidalTimeEmbedding(time_dim)

        # ---------------------------------------------------------
        # Noisy target image pathway
        # ---------------------------------------------------------

        self.input_conv = nn.Conv2d(
            image_channels,
            base_channels,
            kernel_size=3,
            padding=1,
        )

        # ---------------------------------------------------------
        # 256x256 condition pathway
        # ---------------------------------------------------------

        self.condition_256 = AdaptiveConditionBlock(
            condition_channels,
            base_channels,
        )

        # ---------------------------------------------------------
        # 128x128 condition pathway
        # ---------------------------------------------------------

        self.condition_down_128 = nn.Sequential(
            nn.Conv2d(
                base_channels,
                base_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
        )

        self.condition_128 = AdaptiveConditionBlock(
            base_channels,
            base_channels,
        )

        # ---------------------------------------------------------
        # 64x64 condition pathway
        # ---------------------------------------------------------

        self.condition_down_64 = nn.Sequential(
            nn.Conv2d(
                base_channels,
                base_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
        )

        self.condition_64 = AdaptiveConditionBlock(
            base_channels,
            base_channels * 2,
        )

        # ---------------------------------------------------------
        # U-Net encoder
        # ---------------------------------------------------------

        self.down_block1 = ResidualBlock(
            base_channels,
            base_channels,
            time_dim,
        )

        self.downsample1 = nn.Conv2d(
            base_channels,
            base_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.down_block2 = ResidualBlock(
            base_channels,
            base_channels * 2,
            time_dim,
        )

        self.downsample2 = nn.Conv2d(
            base_channels * 2,
            base_channels * 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        # ---------------------------------------------------------
        # U-Net middle
        # ---------------------------------------------------------

        self.middle_block1 = ResidualBlock(
            base_channels * 2,
            base_channels * 4,
            time_dim,
        )

        self.middle_block2 = ResidualBlock(
            base_channels * 4,
            base_channels * 4,
            time_dim,
        )

        # ---------------------------------------------------------
        # U-Net decoder
        # ---------------------------------------------------------

        self.upsample1 = nn.Sequential(
            nn.Upsample(
                scale_factor=2,
                mode="nearest",
            ),
            nn.Conv2d(
                base_channels * 4,
                base_channels * 4,
                kernel_size=3,
                padding=1,
            ),
        )

        self.up_block1 = ResidualBlock(
            base_channels * 4 + base_channels * 2,
            base_channels * 2,
            time_dim,
        )

        self.upsample2 = nn.Sequential(
            nn.Upsample(
                scale_factor=2,
                mode="nearest",
            ),
            nn.Conv2d(
                base_channels * 2,
                base_channels * 2,
                kernel_size=3,
                padding=1,
            ),
        )

        self.up_block2 = ResidualBlock(
            base_channels * 2 + base_channels,
            base_channels,
            time_dim,
        )

        self.output_conv = nn.Conv2d(
            base_channels,
            image_channels,
            kernel_size=3,
            padding=1,
        )

        # Stores the most recent gates for diagnostics.
        # This is useful during architecture verification and later
        # adaptive-gate analysis.
        self.last_gates: dict[str, torch.Tensor] = {}

    def forward(
        self,
        noisy_image: torch.Tensor,
        condition: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:

        # ---------------------------------------------------------
        # Timestep embedding
        # ---------------------------------------------------------

        time_emb = self.time_embedding(timesteps)

        # ---------------------------------------------------------
        # Multi-scale adaptive condition extraction
        # ---------------------------------------------------------

        condition_256, gate_256 = self.condition_256(condition)

        condition_128_input = self.condition_down_128(
            condition_256
        )

        condition_128, gate_128 = self.condition_128(
            condition_128_input
        )

        condition_64_input = self.condition_down_64(
            condition_128
        )

        condition_64, gate_64 = self.condition_64(
            condition_64_input
        )

        # ---------------------------------------------------------
        # Store gates for diagnostics
        # ---------------------------------------------------------

        self.last_gates = {
            "256": gate_256,
            "128": gate_128,
            "64": gate_64,
        }

        # ---------------------------------------------------------
        # U-Net encoder
        # ---------------------------------------------------------

        x = self.input_conv(noisy_image)

        # Adaptive 256x256 conditioning
        x = x + condition_256

        x1 = self.down_block1(
            x,
            time_emb,
        )

        # 256 -> 128
        x = self.downsample1(x1)

        # Adaptive 128x128 conditioning
        x = x + condition_128

        x2 = self.down_block2(
            x,
            time_emb,
        )

        # 128 -> 64
        x = self.downsample2(x2)

        # Adaptive 64x64 conditioning
        x = x + condition_64

        # ---------------------------------------------------------
        # Middle
        # ---------------------------------------------------------

        x = self.middle_block1(
            x,
            time_emb,
        )

        x = self.middle_block2(
            x,
            time_emb,
        )

        # ---------------------------------------------------------
        # Decoder
        # ---------------------------------------------------------

        # 64 -> 128
        x = self.upsample1(x)

        x = torch.cat(
            [x, x2],
            dim=1,
        )

        x = self.up_block1(
            x,
            time_emb,
        )

        # 128 -> 256
        x = self.upsample2(x)

        x = torch.cat(
            [x, x1],
            dim=1,
        )

        x = self.up_block2(
            x,
            time_emb,
        )

        # ---------------------------------------------------------
        # Noise prediction
        # ---------------------------------------------------------

        return self.output_conv(x)
