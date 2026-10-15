from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SinusoidalTimeEmbedding(nn.Module):
    """
    Converts an integer diffusion timestep into a continuous
    sinusoidal embedding.

    Input:
        timesteps: [B]

    Output:
        embedding: [B, embedding_dim]
    """

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be greater than 0")

        self.embedding_dim = embedding_dim

    def forward(self, timesteps: Tensor) -> Tensor:
        half_dim = self.embedding_dim // 2

        exponent = math.log(10000.0) / max(half_dim - 1, 1)

        frequencies = torch.exp(
            torch.arange(
                half_dim,
                device=timesteps.device,
                dtype=torch.float32,
            )
            * -exponent
        )

        arguments = (
            timesteps.float().unsqueeze(1)
            * frequencies.unsqueeze(0)
        )

        embedding = torch.cat(
            [
                torch.sin(arguments),
                torch.cos(arguments),
            ],
            dim=1,
        )

        if self.embedding_dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))

        return embedding


class ResidualBlock(nn.Module):
    """
    Residual convolutional block with timestep conditioning.

    Input:
        x: [B, in_channels, H, W]
        time_embedding: [B, time_dim]

    Output:
        [B, out_channels, H, W]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
    ) -> None:
        super().__init__()

        self.norm1 = nn.GroupNorm(
            num_groups=8,
            num_channels=in_channels,
        )

        self.activation1 = nn.SiLU()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

        self.time_projection = nn.Linear(
            time_dim,
            out_channels,
        )

        self.norm2 = nn.GroupNorm(
            num_groups=8,
            num_channels=out_channels,
        )

        self.activation2 = nn.SiLU()

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

        if in_channels != out_channels:
            self.skip = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
            )
        else:
            self.skip = nn.Identity()

    def forward(
        self,
        x: Tensor,
        time_embedding: Tensor,
    ) -> Tensor:

        residual = self.skip(x)

        x = self.norm1(x)
        x = self.activation1(x)
        x = self.conv1(x)

        time = self.time_projection(time_embedding)
        time = time.unsqueeze(-1).unsqueeze(-1)

        x = x + time

        x = self.norm2(x)
        x = self.activation2(x)
        x = self.conv2(x)

        return x + residual


class Downsample(nn.Module):
    """Downsample spatial resolution by a factor of 2."""

    def __init__(self, channels: int) -> None:
        super().__init__()

        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Upsample spatial resolution by a factor of 2."""

    def __init__(self, channels: int) -> None:
        super().__init__()

        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(
            x,
            scale_factor=2,
            mode="nearest",
        )

        return self.conv(x)


class MultiScaleConditionalUNet(nn.Module):
    """
    Conditional U-Net for low-light image enhancement using
    multi-scale conditioning.

    The low-light condition is encoded into feature maps at:

        256 x 256
        128 x 128
         64 x 64

    These condition features are injected into corresponding
    levels of the U-Net.

    Input:
        noisy_image: [B, 3, H, W]
        condition:   [B, 3, H, W]
        timesteps:   [B]

    Output:
        predicted noise: [B, 3, H, W]
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
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # ---------------------------------------------------------
        # Noisy target input
        # ---------------------------------------------------------

        self.input_conv = nn.Conv2d(
            image_channels,
            base_channels,
            kernel_size=3,
            padding=1,
        )

        # ---------------------------------------------------------
        # Multi-scale condition encoder
        # ---------------------------------------------------------

        self.condition_conv_256 = nn.Conv2d(
            condition_channels,
            base_channels,
            kernel_size=3,
            padding=1,
        )

        self.condition_down_128 = Downsample(
            base_channels,
        )

        self.condition_down_64 = Downsample(
            base_channels,
        )

        # ---------------------------------------------------------
        # Condition projections
        # ---------------------------------------------------------

        # At 128x128 the U-Net feature map entering down_block2
        # has base_channels (32) channels.
        self.condition_projection_128 = nn.Conv2d(
            base_channels,
            base_channels,
            kernel_size=1,
        )

        # After downsample2 the U-Net feature map has
        # base_channels * 2 (64) channels.
        self.condition_projection_64 = nn.Conv2d(
            base_channels,
            base_channels * 2,
            kernel_size=1,
        )

        # ---------------------------------------------------------
        # Encoder
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
        # Bottleneck
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
        # Decoder
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
        # Output
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
    ) -> Tensor:

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
        # Multi-scale condition features
        # ---------------------------------------------------------

        condition_256 = self.condition_conv_256(
            condition
        )

        condition_128 = self.condition_down_128(
            condition_256
        )

        condition_64 = self.condition_down_64(
            condition_128
        )

        condition_128 = self.condition_projection_128(
            condition_128
        )

        condition_64 = self.condition_projection_64(
            condition_64
        )

        # ---------------------------------------------------------
        # Noisy target input
        # ---------------------------------------------------------

        x = self.input_conv(
            noisy_image
        )

        # Full-resolution condition injection.
        x = x + condition_256

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
        # Encoder level 2
        # ---------------------------------------------------------

        x = x + condition_128

        skip2 = self.down_block2(
            x,
            time_embedding,
        )

        x = self.downsample2(
            skip2
        )

        # ---------------------------------------------------------
        # Bottleneck
        # ---------------------------------------------------------

        x = x + condition_64

        x = self.middle_block1(
            x,
            time_embedding,
        )

        x = self.middle_block2(
            x,
            time_embedding,
        )

        # ---------------------------------------------------------
        # Decoder level 1
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

        # ---------------------------------------------------------
        # Decoder level 2
        # ---------------------------------------------------------

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
