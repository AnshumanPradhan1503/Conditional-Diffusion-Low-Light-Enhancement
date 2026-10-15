from __future__ import annotations

import src.train as train_module

from src.configs import multiscale_training_config
from src.models.multiscale_unet import MultiScaleConditionalUNet


# Use the dedicated multi-scale configuration.
train_module.config = multiscale_training_config


def create_multiscale_model(device):
    """Create the multi-scale conditional U-Net."""

    model = MultiScaleConditionalUNet(
        image_channels=train_module.config.IMAGE_CHANNELS,
        condition_channels=train_module.config.CONDITION_CHANNELS,
        base_channels=train_module.config.BASE_CHANNELS,
        time_dim=train_module.config.TIME_EMBEDDING_DIM,
    )

    model = model.to(device)

    return model


# Replace the original model factory used by src.train.
train_module.create_model = create_multiscale_model


if __name__ == "__main__":
    train_module.main()
