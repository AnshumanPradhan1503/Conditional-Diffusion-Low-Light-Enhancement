from src.configs import training_config as config


def main():

    print("=" * 60)
    print("TASK 7 — TRAINING CONFIGURATION VERIFICATION")
    print("=" * 60)

    print("\nProject paths:")
    print(f"  Project root : {config.PROJECT_ROOT}")
    print(f"  Data root    : {config.DATA_ROOT}")
    print(f"  Split root   : {config.SPLIT_ROOT}")
    print(f"  Checkpoints  : {config.CHECKPOINT_ROOT}")
    print(f"  Results      : {config.RESULTS_ROOT}")

    print("\nData configuration:")
    print(f"  Image size   : {config.IMAGE_SIZE}")
    print(f"  Batch size   : {config.BATCH_SIZE}")
    print(f"  Num workers  : {config.NUM_WORKERS}")
    print(f"  Pin memory   : {config.PIN_MEMORY}")

    print("\nDiffusion configuration:")
    print(f"  Timesteps    : {config.TIMESTEPS}")
    print(f"  Beta start   : {config.BETA_START}")
    print(f"  Beta end     : {config.BETA_END}")

    print("\nModel configuration:")
    print(f"  Image channels     : {config.IMAGE_CHANNELS}")
    print(f"  Condition channels : {config.CONDITION_CHANNELS}")
    print(f"  Base channels      : {config.BASE_CHANNELS}")
    print(f"  Time embedding     : {config.TIME_EMBEDDING_DIM}")

    print("\nOptimization:")
    print(f"  Learning rate : {config.LEARNING_RATE}")
    print(f"  Weight decay  : {config.WEIGHT_DECAY}")

    print("\nTraining:")
    print(f"  Epochs        : {config.EPOCHS}")
    print(f"  Gradient clip : {config.GRADIENT_CLIP_MAX_NORM}")

    print("\nCheckpointing:")
    print(f"  Save every    : {config.SAVE_EVERY_EPOCH} epochs")

    print("\nValidation:")
    print(f"  Validate every: {config.VALIDATE_EVERY_EPOCH} epoch")

    print("\nDevice:")
    print(f"  {config.DEVICE}")

    # --------------------------------------------------------
    # Assertions
    # --------------------------------------------------------

    assert config.IMAGE_SIZE == 256
    assert config.BATCH_SIZE >= 1

    assert config.TIMESTEPS == 1000
    assert 0 < config.BETA_START < config.BETA_END < 1

    assert config.IMAGE_CHANNELS == 3
    assert config.CONDITION_CHANNELS == 3

    assert config.BASE_CHANNELS > 0
    assert config.TIME_EMBEDDING_DIM > 0

    assert config.LEARNING_RATE > 0
    assert config.WEIGHT_DECAY >= 0

    assert config.EPOCHS > 0
    assert config.GRADIENT_CLIP_MAX_NORM > 0

    assert config.RANDOM_SEED == 42

    print("\nVerification:")
    print("  [PASS] Image configuration")
    print("  [PASS] Diffusion configuration")
    print("  [PASS] Model configuration")
    print("  [PASS] Optimization configuration")
    print("  [PASS] Training configuration")
    print("  [PASS] Reproducibility configuration")

    print("\n" + "=" * 60)
    print("TASK 7 TRAINING CONFIGURATION VERIFICATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()