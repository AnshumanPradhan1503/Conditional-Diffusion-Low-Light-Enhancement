from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.unet import SinusoidalTimeEmbedding


class IlluminationEncoder(nn.Module):
    """
    Encodes the low-frequency illumination proxy at multiple
    spatial resolutions.

    Input:
        [B, 1, 256, 256]

    Outputs:
        256x256 -> [B, 32, 256, 256]
        128x128 -> [B, 64, 128, 128]
        64x64   -> [B, 128, 64, 64]
    """

    def __init__(
        self,
        base_channels: int = 32,
    ):
        super().__init__()

        self.block_256 = nn.Sequential(
            nn.Conv2d(
                1,
                base_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.GroupNorm(
                8,
                base_channels,
            ),
            nn.SiLU(),
            nn.Conv2d(
                base_channels,
                base_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.GroupNorm(
                8,
                base_channels,
            ),
            nn.SiLU(),
        )

        self.block_128 = nn.Sequential(
            nn.Conv2d(
                base_channels,
                base_channels * 2,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(
                8,
                base_channels * 2,
            ),
            nn.SiLU(),
            nn.Conv2d(
                base_channels * 2,
                base_channels * 2,
                kernel_size=3,
                padding=1,
            ),
            nn.GroupNorm(
                8,
                base_channels * 2,
            ),
            nn.SiLU(),
        )

        self.block_64 = nn.Sequential(
            nn.Conv2d(
                base_channels * 2,
                base_channels * 4,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(
                8,
                base_channels * 4,
            ),
            nn.SiLU(),
            nn.Conv2d(
                base_channels * 4,
                base_channels * 4,
                kernel_size=3,
                padding=1,
            ),
            nn.GroupNorm(
                8,
                base_channels * 4,
            ),
            nn.SiLU(),
        )

    def forward(
        self,
        illumination: torch.Tensor,
    ):
        illumination_256 = self.block_256(
            illumination
        )

        illumination_128 = self.block_128(
            illumination_256
        )

        illumination_64 = self.block_64(
            illumination_128
        )

        return (
            illumination_256,
            illumination_128,
            illumination_64,
        )


class IlluminationAwareConditionalUNet(nn.Module):
    """
    Conditional U-Net with a separate low-frequency
    illumination-conditioning pathway.

    Main condition:
        Low-light RGB image [B, 3, H, W]

    Additional condition:
        Low-frequency illumination proxy [B, 1, H, W]

    The illumination pathway is injected at multiple
    spatial resolutions.
    """

    def __init__(
        self,
        image_channels: int = 3,
        condition_channels: int = 3,
        base_channels: int = 32,
        time_dim: int = 128,
    ):
        super().__init__()

        self.image_channels = image_channels
        self.condition_channels = condition_channels
        self.base_channels = base_channels
        self.time_dim = time_dim

        # -------------------------------------------------
        # Time embedding
        # -------------------------------------------------

        self.time_embedding = SinusoidalTimeEmbedding(
            time_dim
        )

        # -------------------------------------------------
        # Main noisy-image + RGB condition pathway
        # -------------------------------------------------

        self.input_conv = nn.Conv2d(
            image_channels + condition_channels,
            base_channels,
            kernel_size=3,
            padding=1,
        )

        self.down_256 = nn.Sequential(
            nn.GroupNorm(
                8,
                base_channels,
            ),
            nn.SiLU(),
            nn.Conv2d(
                base_channels,
                base_channels,
                kernel_size=3,
                padding=1,
            ),
        )

        self.downsample_128 = nn.Conv2d(
            base_channels,
            base_channels * 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.down_128 = nn.Sequential(
            nn.GroupNorm(
                8,
                base_channels * 2,
            ),
            nn.SiLU(),
            nn.Conv2d(
                base_channels * 2,
                base_channels * 2,
                kernel_size=3,
                padding=1,
            ),
        )

        self.downsample_64 = nn.Conv2d(
            base_channels * 2,
            base_channels * 4,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.down_64 = nn.Sequential(
            nn.GroupNorm(
                8,
                base_channels * 4,
            ),
            nn.SiLU(),
            nn.Conv2d(
                base_channels * 4,
                base_channels * 4,
                kernel_size=3,
                padding=1,
            ),
        )

        # -------------------------------------------------
        # Bottleneck
        # -------------------------------------------------

        self.bottleneck = nn.Sequential(
            nn.GroupNorm(
                8,
                base_channels * 4,
            ),
            nn.SiLU(),
            nn.Conv2d(
                base_channels * 4,
                base_channels * 4,
                kernel_size=3,
                padding=1,
            ),
        )

        self.time_projection = nn.Linear(
            time_dim,
            base_channels * 4,
        )

        # -------------------------------------------------
        # Separate illumination pathway
        # -------------------------------------------------

        self.illumination_encoder = IlluminationEncoder(
            base_channels=base_channels
        )

        # -------------------------------------------------
        # Illumination projection layers
        # -------------------------------------------------

        self.illumination_proj_256 = nn.Conv2d(
            base_channels,
            base_channels,
            kernel_size=1,
        )

        self.illumination_proj_128 = nn.Conv2d(
            base_channels * 2,
            base_channels * 2,
            kernel_size=1,
        )

        self.illumination_proj_64 = nn.Conv2d(
            base_channels * 4,
            base_channels * 4,
            kernel_size=1,
        )

        # -------------------------------------------------
        # Upsampling pathway
        # -------------------------------------------------

        self.up_128 = nn.Sequential(
            nn.GroupNorm(
                8,
                base_channels * 4,
            ),
            nn.SiLU(),
            nn.Conv2d(
                base_channels * 4,
                base_channels * 2,
                kernel_size=3,
                padding=1,
            ),
        )

        self.up_256 = nn.Sequential(
            nn.GroupNorm(
                8,
                base_channels * 2,
            ),
            nn.SiLU(),
            nn.Conv2d(
                base_channels * 2,
                base_channels,
                kernel_size=3,
                padding=1,
            ),
        )

        self.output_norm = nn.GroupNorm(
            8,
            base_channels,
        )

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
        illumination: torch.Tensor,
    ) -> torch.Tensor:

        # -------------------------------------------------
        # Validate inputs
        # -------------------------------------------------

        if noisy_image.ndim != 4:
            raise ValueError(
                "noisy_image must have shape [B, C, H, W]."
            )

        if condition.ndim != 4:
            raise ValueError(
                "condition must have shape [B, C, H, W]."
            )

        if illumination.ndim != 4:
            raise ValueError(
                "illumination must have shape [B, 1, H, W]."
            )

        if illumination.shape[1] != 1:
            raise ValueError(
                "illumination must contain exactly one channel."
            )

        # -------------------------------------------------
        # Time embedding
        # -------------------------------------------------

        time_embedding = self.time_embedding(
            timesteps
        )

        # -------------------------------------------------
        # Main RGB conditioning pathway
        # -------------------------------------------------

        x = torch.cat(
            [
                noisy_image,
                condition,
            ],
            dim=1,
        )

        x = self.input_conv(x)

        # 256x256
        x_256 = self.down_256(x)

        # -------------------------------------------------
        # Illumination pathway
        # -------------------------------------------------

        illumination_256, illumination_128, illumination_64 = (
            self.illumination_encoder(
                illumination
            )
        )

        illumination_256 = self.illumination_proj_256(
            illumination_256
        )

        illumination_128 = self.illumination_proj_128(
            illumination_128
        )

        illumination_64 = self.illumination_proj_64(
            illumination_64
        )

        # Separate illumination information is added
        # to the corresponding spatial feature scale.
        x_256 = x_256 + illumination_256

        # -------------------------------------------------
        # Downsample to 128x128
        # -------------------------------------------------

        x_128 = self.downsample_128(
            x_256
        )

        x_128 = self.down_128(
            x_128
        )

        x_128 = x_128 + illumination_128

        # -------------------------------------------------
        # Downsample to 64x64
        # -------------------------------------------------

        x_64 = self.downsample_64(
            x_128
        )

        x_64 = self.down_64(
            x_64
        )

        x_64 = x_64 + illumination_64

        # -------------------------------------------------
        # Bottleneck + timestep conditioning
        # -------------------------------------------------

        x = self.bottleneck(
            x_64
        )

        time_features = self.time_projection(
            time_embedding
        )

        time_features = time_features[
            :, :, None, None
        ]

        x = x + time_features

        # -------------------------------------------------
        # Up to 128x128
        # -------------------------------------------------

        x = F.interpolate(
            x,
            size=x_128.shape[-2:],
            mode="nearest",
        )

        x = self.up_128(
            x
        )

        # -------------------------------------------------
        # Up to 256x256
        # -------------------------------------------------

        x = F.interpolate(
            x,
            size=x_256.shape[-2:],
            mode="nearest",
        )

        x = self.up_256(
            x
        )

        # -------------------------------------------------
        # Output
        # -------------------------------------------------

        x = self.output_norm(x)
        x = F.silu(x)

        return self.output_conv(x)