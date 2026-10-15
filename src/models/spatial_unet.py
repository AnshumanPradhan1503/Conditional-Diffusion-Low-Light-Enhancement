from __future__ import annotations

import torch
import torch.nn as nn

from src.models.unet import (
    SinusoidalTimeEmbedding,
    ResidualBlock,
)


class SpatialConditionBlock(nn.Module):
    """
    Extract spatial features from the low-light condition.

    The output remains spatially aligned with the corresponding
    U-Net feature resolution.
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
            nn.GroupNorm(
                8,
                feature_channels,
            ),
            nn.SiLU(),
            nn.Conv2d(
                feature_channels,
                feature_channels,
                kernel_size=3,
                padding=1,
            ),
        )

    def forward(
        self,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        return self.encoder(condition)


class SpatialConditionalUNet(nn.Module):
    """
    Experimental conditional DDPM U-Net using stronger
    spatial conditioning.

    The low-light image is encoded into spatial feature maps
    at multiple resolutions and injected into the U-Net:

        256x256
        128x128
         64x64

    The model predicts diffusion noise epsilon.
    """

    def __init__(
        self,
        image_channels: int = 3,
        condition_channels: int = 3,
        base_channels: int = 32,
        time_dim: int = 128,
    ) -> None:
        super().__init__()

        # =========================================================
        # TIME EMBEDDING
        # =========================================================

        self.time_embedding = SinusoidalTimeEmbedding(
            time_dim
        )

        # =========================================================
        # NOISY TARGET BRANCH
        # =========================================================

        self.input_conv = nn.Conv2d(
            image_channels,
            base_channels,
            kernel_size=3,
            padding=1,
        )

        # =========================================================
        # SPATIAL CONDITION ENCODER
        # =========================================================

        # 256 x 256
        self.condition_256 = SpatialConditionBlock(
            condition_channels,
            base_channels,
        )

        # 256 x 256 -> 128 x 128
        self.condition_down_128 = nn.Sequential(
            nn.Conv2d(
                base_channels,
                base_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(
                8,
                base_channels,
            ),
            nn.SiLU(),
        )

        # Additional 128 x 128 processing
        self.condition_128 = SpatialConditionBlock(
            base_channels,
            base_channels,
        )

        # 128 x 128 -> 64 x 64
        self.condition_down_64 = nn.Sequential(
            nn.Conv2d(
                base_channels,
                base_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(
                8,
                base_channels,
            ),
            nn.SiLU(),
        )

        # 64 x 64
        self.condition_64 = SpatialConditionBlock(
            base_channels,
            base_channels * 2,
        )

        # =========================================================
        # U-NET ENCODER
        # =========================================================

        # 256 x 256
        self.down_block1 = ResidualBlock(
            base_channels,
            base_channels,
            time_dim,
        )

        # 256 -> 128
        self.downsample1 = nn.Conv2d(
            base_channels,
            base_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        # 128 x 128
        self.down_block2 = ResidualBlock(
            base_channels,
            base_channels * 2,
            time_dim,
        )

        # 128 -> 64
        self.downsample2 = nn.Conv2d(
            base_channels * 2,
            base_channels * 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        # =========================================================
        # MIDDLE
        # =========================================================

        # 64 x 64
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

        # =========================================================
        # DECODER
        # =========================================================

        # 64 -> 128
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

        # 128 x 128
        # 128 channels from decoder + 64 channels from skip
        self.up_block1 = ResidualBlock(
            base_channels * 4 + base_channels * 2,
            base_channels * 2,
            time_dim,
        )

        # 128 -> 256
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

        # 256 x 256
        # 64 channels from decoder + 32 channels from skip
        self.up_block2 = ResidualBlock(
            base_channels * 2 + base_channels,
            base_channels,
            time_dim,
        )

        # =========================================================
        # OUTPUT
        # =========================================================

        self.output_conv = nn.Conv2d(
            base_channels,
            image_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(
        self,
        noisy_image: torch.Tensor,
        condition: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:

        # =========================================================
        # TIME EMBEDDING
        # =========================================================

        time_emb = self.time_embedding(
            timesteps
        )

        # =========================================================
        # CONDITION FEATURES
        # =========================================================

        # 256 x 256
        condition_256 = self.condition_256(
            condition
        )

        # 256 -> 128
        condition_128_input = self.condition_down_128(
            condition_256
        )

        # 128 x 128
        condition_128 = self.condition_128(
            condition_128_input
        )

        # 128 -> 64
        condition_64_input = self.condition_down_64(
            condition_128
        )

        # 64 x 64
        condition_64 = self.condition_64(
            condition_64_input
        )

        # =========================================================
        # U-NET ENCODER
        # =========================================================

        # Initial noisy-image features
        x = self.input_conv(
            noisy_image
        )

        # ---------------------------------------------------------
        # 256 x 256 spatial conditioning
        # ---------------------------------------------------------

        x = x + condition_256

        x1 = self.down_block1(
            x,
            time_emb,
        )

        # ---------------------------------------------------------
        # Downsample: 256 -> 128
        # ---------------------------------------------------------

        x = self.downsample1(
            x1
        )

        # ---------------------------------------------------------
        # 128 x 128 spatial conditioning
        # ---------------------------------------------------------

        x = x + condition_128

        x2 = self.down_block2(
            x,
            time_emb,
        )

        # ---------------------------------------------------------
        # Downsample: 128 -> 64
        # ---------------------------------------------------------

        x = self.downsample2(
            x2
        )

        # ---------------------------------------------------------
        # 64 x 64 spatial conditioning
        # ---------------------------------------------------------

        x = x + condition_64

        # =========================================================
        # MIDDLE
        # =========================================================

        x = self.middle_block1(
            x,
            time_emb,
        )

        x = self.middle_block2(
            x,
            time_emb,
        )

        # =========================================================
        # DECODER
        # =========================================================

        # ---------------------------------------------------------
        # 64 -> 128
        # ---------------------------------------------------------

        x = self.upsample1(
            x
        )

        x = torch.cat(
            [x, x2],
            dim=1,
        )

        x = self.up_block1(
            x,
            time_emb,
        )

        # ---------------------------------------------------------
        # 128 -> 256
        # ---------------------------------------------------------

        x = self.upsample2(
            x
        )

        x = torch.cat(
            [x, x1],
            dim=1,
        )

        x = self.up_block2(
            x,
            time_emb,
        )

        # =========================================================
        # OUTPUT
        # =========================================================

        return self.output_conv(
            x
        )