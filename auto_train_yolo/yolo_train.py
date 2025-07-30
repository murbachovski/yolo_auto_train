# auto_train_yolo/yolo_train.py

import argparse
from ultralytics import YOLO

def train_model(data_yaml_path, pretrained_model_path, epochs=50, batch=16, imgsz=640, save_dir="models/retrained"):
    model = YOLO(pretrained_model_path)
    model.train(data=data_yaml_path,
                epochs=epochs,
                batch=batch,
                imgsz=imgsz,
                project=save_dir,
                name="retrained",
                exist_ok=True)
    best_weights_path = f"{save_dir}/retrained/weights/best.pt"
    return best_weights_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO 모델 재학습")
    parser.add_argument("--epochs", type=int, default=1, help="학습 epoch 수")
    parser.add_argument("--batch", type=int, default=16, help="배치 사이즈")
    parser.add_argument("--imgsz", type=int, default=640, help="이미지 사이즈")
    parser.add_argument("--data", type=str, required=True, help="data.yaml 경로")
    parser.add_argument("--pretrained", type=str, required=True, help="사전학습 모델 경로")
    args = parser.parse_args()

    train_model(args.data, args.pretrained, epochs=args.epochs, batch=args.batch, imgsz=args.imgsz)
