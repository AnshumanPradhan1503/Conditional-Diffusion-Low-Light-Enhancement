from __future__ import annotations

import src.train as train_module
from src.configs import spatial_training_config
from src.models.spatial_unet import SpatialConditionalUNet


# Use the spatial-conditioning experiment configuration.
train_module.config = spatial_training_config


def create_spatial_model(device):
    model = SpatialConditionalUNet(
        image_channels=train_module.config.IMAGE_CHANNELS,
        condition_channels=train_module.config.CONDITION_CHANNELS,
        base_channels=train_module.config.BASE_CHANNELS,
        time_dim=train_module.config.TIME_EMBEDDING_DIM,
    )

    model = model.to(device)

    return model


# Replace the common training pipeline's model factory.
train_module.create_model = create_spatial_model


if __name__ == "__main__":
    train_module.main()