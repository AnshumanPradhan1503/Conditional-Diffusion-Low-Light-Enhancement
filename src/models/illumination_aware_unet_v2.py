from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from src.models.unet import (
    ResidualBlock,
    Downsample,
    SinusoidalTimeEmbedding,
    Upsample,
)


class IlluminationEncoder(nn.Module):
    """
    Encodes a low-frequency illumination proxy at three spatial scales.

    Input:
        illumination: [B, 1, H, W]

    Outputs:
        illumination_256: [B, base_channels, H, W]
        illumination_128: [B, base_channels, H/2, W/2]
        illumination_64:  [B, base_channels*2, H/4, W/4]
    """

    def __init__(
        self,
        base_channels: int = 32,
    ) -> None:
        super().__init__()

        if base_channels % 8 != 0:
            raise ValueError(
                "base_channels must be divisible by 8"
            )

        # ---------------------------------------------------------
        # 256 x 256 illumination features
        # ---------------------------------------------------------

        self.conv_256 = nn.Sequential(
            nn.Conv2d(
                1,
                base_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.GroupNorm(
                num_groups=8,
                num_channels=base_channels,
            ),
            nn.SiLU(),
            nn.Conv2d(
                base_channels,
                base_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.GroupNorm(
                num_groups=8,
                num_channels=base_channels,
            ),
            nn.SiLU(),
        )

        # ---------------------------------------------------------
        # 128 x 128 illumination features
        # ---------------------------------------------------------

        self.down_128 = nn.Conv2d(
            base_channels,
            base_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.conv_128 = nn.Sequential(
            nn.GroupNorm(
                num_groups=8,
                num_channels=base_channels,
            ),
            nn.SiLU(),
            nn.Conv2d(
                base_channels,
                base_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.GroupNorm(
                num_groups=8,
                num_channels=base_channels,
            ),
            nn.SiLU(),
        )

        # ---------------------------------------------------------
        # 64 x 64 illumination features
        # ---------------------------------------------------------

        self.down_64 = nn.Conv2d(
            base_channels,
            base_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.conv_64 = nn.Sequential(
            nn.GroupNorm(
                num_groups=8,
                num_channels=base_channels,
            ),
            nn.SiLU(),
            nn.Conv2d(
                base_channels,
                base_channels * 2,
                kernel_size=3,
                padding=1,
            ),
            nn.GroupNorm(
                num_groups=8,
                num_channels=base_channels * 2,
            ),
            nn.SiLU(),
        )

        # ---------------------------------------------------------
        # Near-zero output projections
        #
        # These make the illumination pathway initially contribute
        # approximately zero to the original U-Net features.
        # ---------------------------------------------------------

        self.projection_256 = nn.Conv2d(
            base_channels,
            base_channels,
            kernel_size=1,
        )

        self.projection_128 = nn.Conv2d(
            base_channels,
            base_channels,
            kernel_size=1,
        )

        self.projection_64 = nn.Conv2d(
            base_channels * 2,
            base_channels * 2,
            kernel_size=1,
        )

        nn.init.zeros_(self.projection_256.weight)
        nn.init.zeros_(self.projection_256.bias)

        nn.init.zeros_(self.projection_128.weight)
        nn.init.zeros_(self.projection_128.bias)

        nn.init.zeros_(self.projection_64.weight)
        nn.init.zeros_(self.projection_64.bias)

    def forward(
        self,
        illumination: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Encode illumination information at three scales.
        """

        if illumination.ndim != 4:
            raise ValueError(
                "illumination must have shape [B, 1, H, W]"
            )

        if illumination.shape[1] != 1:
            raise ValueError(
                "illumination must have exactly 1 channel"
            )

        illumination_256 = self.conv_256(
            illumination
        )

        illumination_128 = self.down_128(
            illumination_256
        )

        illumination_128 = self.conv_128(
            illumination_128
        )

        illumination_64 = self.down_64(
            illumination_128
        )

        illumination_64 = self.conv_64(
            illumination_64
        )

        illumination_256 = self.projection_256(
            illumination_256
        )

        illumination_128 = self.projection_128(
            illumination_128
        )

        illumination_64 = self.projection_64(
            illumination_64
        )

        return (
            illumination_256,
            illumination_128,
            illumination_64,
        )


class IlluminationAwareConditionalUNet(nn.Module):
    """
    Illumination-aware conditional U-Net.

    This model preserves the original ConditionalUNet architecture
    and adds a separate low-frequency illumination pathway.

    Inputs:
        noisy_image:
            [B, image_channels, H, W]

        condition:
            [B, condition_channels, H, W]

        timesteps:
            [B]

        illumination:
            [B, 1, H, W]

    Output:
        predicted noise:
            [B, image_channels, H, W]
    """

    def __init__(
        self,
        image_channels: int = 3,
        condition_channels: int = 3,
        base_channels: int = 32,
        time_dim: int = 128,
    ) -> None:
        super().__init__()

        if base_channels % 8 != 0:
            raise ValueError(
                "base_channels must be divisible by 8"
            )

        self.image_channels = image_channels
        self.condition_channels = condition_channels
        self.base_channels = base_channels
        self.time_dim = time_dim

        # ---------------------------------------------------------
        # Time embedding
        # ---------------------------------------------------------

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(
                time_dim,
                time_dim,
            ),
            nn.SiLU(),
            nn.Linear(
                time_dim,
                time_dim,
            ),
        )

        # ---------------------------------------------------------
        # Original U-Net input
        #
        # Noisy target + low-light condition
        # ---------------------------------------------------------

        self.input_conv = nn.Conv2d(
            image_channels + condition_channels,
            base_channels,
            kernel_size=3,
            padding=1,
        )

        # ---------------------------------------------------------
        # NEW: separate illumination encoder
        # ---------------------------------------------------------

        self.illumination_encoder = IlluminationEncoder(
            base_channels=base_channels,
        )

        # ---------------------------------------------------------
        # Original encoder
        # ---------------------------------------------------------

        self.down_block1 = ResidualBlock(
            base_channels,
            base_channels,
            time_dim,
        )

        self.downsample1 = Downsample(
            base_channels,
        )

        self.down_block2 = ResidualBlock(
            base_channels,
            base_channels * 2,
            time_dim,
        )

        self.downsample2 = Downsample(
            base_channels * 2,
        )

        # ---------------------------------------------------------
        # Original bottleneck
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
        # Original decoder
        # ---------------------------------------------------------

        self.upsample1 = Upsample(
            base_channels * 4,
        )

        self.up_block1 = ResidualBlock(
            base_channels * 4 + base_channels * 2,
            base_channels * 2,
            time_dim,
        )

        self.upsample2 = Upsample(
            base_channels * 2,
        )

        self.up_block2 = ResidualBlock(
            base_channels * 2 + base_channels,
            base_channels,
            time_dim,
        )

        # ---------------------------------------------------------
        # Original output
        # ---------------------------------------------------------

        self.output_norm = nn.GroupNorm(
            num_groups=8,
            num_channels=base_channels,
        )

        self.output_activation = nn.SiLU()

        self.output_conv = nn.Conv2d(
            base_channels,
            image_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(
        self,
        noisy_image: Tensor,
        condition: Tensor,
        timesteps: Tensor,
        illumination: Tensor,
    ) -> Tensor:
        """
        Forward pass.

        The original U-Net computation is preserved.
        Illumination information is added at the three encoder
        resolutions.
        """

        # ---------------------------------------------------------
        # Validate inputs
        # ---------------------------------------------------------

        if noisy_image.ndim != 4:
            raise ValueError(
                "noisy_image must have shape [B, C, H, W]"
            )

        if condition.ndim != 4:
            raise ValueError(
                "condition must have shape [B, C, H, W]"
            )

        if illumination.ndim != 4:
            raise ValueError(
                "illumination must have shape [B, 1, H, W]"
            )

        if noisy_image.shape != condition.shape:
            raise ValueError(
                "noisy_image and condition must have the same shape"
            )

        if noisy_image.shape[1] != self.image_channels:
            raise ValueError(
                f"Expected noisy_image to have "
                f"{self.image_channels} channels"
            )

        if condition.shape[1] != self.condition_channels:
            raise ValueError(
                f"Expected condition to have "
                f"{self.condition_channels} channels"
            )

        if illumination.shape[1] != 1:
            raise ValueError(
                "illumination must have exactly 1 channel"
            )

        if illumination.shape[0] != noisy_image.shape[0]:
            raise ValueError(
                "Batch size of illumination must match images"
            )

        if illumination.shape[2:] != noisy_image.shape[2:]:
            raise ValueError(
                "Spatial dimensions of illumination must match images"
            )

        if timesteps.ndim != 1:
            raise ValueError(
                "timesteps must have shape [B]"
            )

        if timesteps.shape[0] != noisy_image.shape[0]:
            raise ValueError(
                "Batch size of timesteps must match images"
            )

        # ---------------------------------------------------------
        # Time embedding
        # ---------------------------------------------------------

        time_embedding = self.time_embedding(
            timesteps
        )

        # ---------------------------------------------------------
        # NEW: illumination pathway
        # ---------------------------------------------------------

        (
            illumination_256,
            illumination_128,
            illumination_64,
        ) = self.illumination_encoder(
            illumination
        )

        # ---------------------------------------------------------
        # ORIGINAL U-Net:
        # Concatenate noisy target + low-light condition
        # ---------------------------------------------------------

        x = torch.cat(
            [
                noisy_image,
                condition,
            ],
            dim=1,
        )

        # ---------------------------------------------------------
        # Original input
        # ---------------------------------------------------------

        x = self.input_conv(x)

        # ---------------------------------------------------------
        # 256 x 256
        #
        # Original feature + illumination feature
        # ---------------------------------------------------------

        x = x + illumination_256

        # ---------------------------------------------------------
        # Encoder level 1
        # ---------------------------------------------------------

        skip1 = self.down_block1(
            x,
            time_embedding,
        )

        x = self.downsample1(
            skip1
        )

        # ---------------------------------------------------------
        # 128 x 128
        #
        # Original feature + illumination feature
        # ---------------------------------------------------------

        x = x + illumination_128

        # ---------------------------------------------------------
        # Encoder level 2
        # ---------------------------------------------------------

        skip2 = self.down_block2(
            x,
            time_embedding,
        )

        x = self.downsample2(
            skip2
        )

        # ---------------------------------------------------------
        # 64 x 64
        #
        # Original feature + illumination feature
        # ---------------------------------------------------------

        x = x + illumination_64

        # ---------------------------------------------------------
        # Bottleneck
        # ---------------------------------------------------------

        x = self.middle_block1(
            x,
            time_embedding,
        )

        x = self.middle_block2(
            x,
            time_embedding,
        )

        # ---------------------------------------------------------
        # Decoder
        # ---------------------------------------------------------

        x = self.upsample1(x)

        x = torch.cat(
            [
                x,
                skip2,
            ],
            dim=1,
        )

        x = self.up_block1(
            x,
            time_embedding,
        )

        x = self.upsample2(x)

        x = torch.cat(
            [
                x,
                skip1,
            ],
            dim=1,
        )

        x = self.up_block2(
            x,
            time_embedding,
        )

        # ---------------------------------------------------------
        # Output
        # ---------------------------------------------------------

        x = self.output_norm(x)
        x = self.output_activation(x)
        x = self.output_conv(x)

        return x