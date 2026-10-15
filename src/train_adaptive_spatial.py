from __future__ import annotations

import torch
import src.train as train_module

from src.models.adaptive_spatial_unet import (
    AdaptiveSpatialConditionalUNet,
)

from src.configs import adaptive_spatial_training_config


train_module.config = adaptive_spatial_training_config


def create_model(device: torch.device):
    model = AdaptiveSpatialConditionalUNet(
        image_channels=adaptive_spatial_training_config.IMAGE_CHANNELS,
        condition_channels=adaptive_spatial_training_config.CONDITION_CHANNELS,
        base_channels=adaptive_spatial_training_config.BASE_CHANNELS,
        time_dim=adaptive_spatial_training_config.TIME_EMBEDDING_DIM,
    )

    return model.to(device)


train_module.create_model = create_model


if __name__ == "__main__":
    train_module.main()