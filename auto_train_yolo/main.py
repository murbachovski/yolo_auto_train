# auto_train_yolo/main.py

from yolo_pipeline import run_loop_pipeline
from utils.config_loader import load_config

if __name__ == "__main__":
    config = load_config("auto_train_yolo/utils/config.yaml")
    run_loop_pipeline(config)
