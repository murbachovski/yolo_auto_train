# auto_train_yolo/utils/config_loader.py

import yaml

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config
