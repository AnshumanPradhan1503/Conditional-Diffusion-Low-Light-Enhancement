import src.train as train_module

from src.configs import capacity_48_config

# Replace the configuration used by the training module
train_module.config = capacity_48_config

if __name__ == "__main__":
    train_module.main()