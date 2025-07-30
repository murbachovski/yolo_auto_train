# auto_train_yolo/yolo_create_data_yaml.py

import os
import yaml

def create_data_yaml(base_dir, detected_class_ids=None, class_id_to_name=None):
    data_yaml_path = os.path.join(base_dir, "data.yaml")

    train_path = os.path.abspath(os.path.join(base_dir, "train", "images"))
    val_path = os.path.abspath(os.path.join(base_dir, "valid", "images"))
    test_path = os.path.abspath(os.path.join(base_dir, "test", "images"))

    if detected_class_ids is not None and class_id_to_name is not None:
        classes = [class_id_to_name[i] for i in detected_class_ids if i in class_id_to_name]
    else:
        classes = ["person"]  # fallback

    data = {
        "train": train_path,
        "val": val_path,
        "test": test_path,
        "nc": len(classes),
        "names": classes
    }

    with open(data_yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

    print(f"✅ data.yaml 생성 완료: {data_yaml_path}")
    print(f" - 클래스 수: {len(classes)}")
    print(f" - 클래스명: {classes}")

    return data_yaml_path
