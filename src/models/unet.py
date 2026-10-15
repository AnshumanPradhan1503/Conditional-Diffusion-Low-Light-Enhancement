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
        t: [B]

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

        arguments = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)

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


class ConditionalUNet(nn.Module):
    """
    Conditional U-Net for low-light image enhancement.

    The model receives:

        noisy target image : 3 channels
        low-light condition: 3 channels

    These are concatenated into a 6-channel input.

    The network additionally receives the diffusion timestep.

    Output:
        predicted noise with 3 channels.
    """

    def __init__(
        self,
        image_channels: int = 3,
        condition_channels: int = 3,
        base_channels: int = 32,
        time_dim: int = 128,
    ) -> None:
        super().__init__()

        self.image_channels = image_channels
        self.condition_channels = condition_channels
        self.base_channels = base_channels
        self.time_dim = time_dim

        # -----------------------------------------------------
        # Time embedding
        # -----------------------------------------------------

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # -----------------------------------------------------
        # Input
        # -----------------------------------------------------

        self.input_conv = nn.Conv2d(
            image_channels + condition_channels,
            base_channels,
            kernel_size=3,
            padding=1,
        )

        # -----------------------------------------------------
        # Encoder
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # Bottleneck
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # Decoder
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # Output
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # Validate inputs
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # Time embedding
        # -----------------------------------------------------

        time_embedding = self.time_embedding(timesteps)

        # -----------------------------------------------------
        # Concatenate noisy target + low-light condition
        # -----------------------------------------------------

        x = torch.cat(
            [
                noisy_image,
                condition,
            ],
            dim=1,
        )

        # -----------------------------------------------------
        # Input
        # -----------------------------------------------------

        x = self.input_conv(x)

        # -----------------------------------------------------
        # Encoder
        # -----------------------------------------------------

        skip1 = self.down_block1(
            x,
            time_embedding,
        )

        x = self.downsample1(skip1)

        skip2 = self.down_block2(
            x,
            time_embedding,
        )

        x = self.downsample2(skip2)

        # -----------------------------------------------------
        # Bottleneck
        # -----------------------------------------------------

        x = self.middle_block1(
            x,
            time_embedding,
        )

        x = self.middle_block2(
            x,
            time_embedding,
        )

        # -----------------------------------------------------
        # Decoder
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # Output
        # -----------------------------------------------------

        x = self.output_norm(x)
        x = self.output_activation(x)
        x = self.output_conv(x)

        return x