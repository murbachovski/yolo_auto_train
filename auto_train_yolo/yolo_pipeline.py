# auto_train_yolo/yolo_pipeline.py

from yolo_create_data_yaml import create_data_yaml
from yolo_train import train_model
from yolo_eval import evaluate_model
from yolo_preprocessing import collect_and_label_frames
from ultralytics import YOLO

def run_loop_pipeline(config):
    iteration = 1
    # while True:
        # print(f"===== {iteration} 번째 반복 시작 =====")

    save_root, detected_ids, id2name = collect_and_label_frames(
        model_path=config["train"]["pretrained_model_path"],
        max_frames=config["camera"]["max_frames"],
        camera_id=config["camera"]["camera_id"]
    )

    data_yaml_path = create_data_yaml(
        save_root,
        detected_class_ids=detected_ids,
        class_id_to_name=id2name
    )

    best_model_path = train_model(
        data_yaml_path,
        config["train"]["pretrained_model_path"],
        epochs=config["train"]["epochs"],
        batch=config["train"]["batch_size"],
        imgsz=config["train"]["imgsz"],
        save_dir=config["train"]["save_dir"]
    )

    new_model = YOLO(best_model_path)
    accuracy = evaluate_model(new_model, data_yaml_path)
    print(f"===== {iteration} 번째 반복 종료 - mAP50: {accuracy} =====\n")

    # config["train"]["pretrained_model_path"] = best_model_path
    # iteration += 1
