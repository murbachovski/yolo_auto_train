# auto_train_yolo/yolo_preprocessing.py

from ultralytics import YOLO
import cv2
import os
from datetime import datetime

def init_dataset_folders(base_path):
    folders = {
        "train": os.path.join(base_path, "train"),
        "valid": os.path.join(base_path, "valid"),
        "test": os.path.join(base_path, "test"),
    }
    for folder in folders.values():
        os.makedirs(os.path.join(folder, "images"), exist_ok=True)
        os.makedirs(os.path.join(folder, "labels"), exist_ok=True)
    return folders

def get_split_type(frame_count, split_counts=(7, 9, 10)):
# def get_split_type(frame_count, conifg):
    # train_max, valid_max, test_max = conifg["data"]["train_data_size"], conifg["data"]["valid_data_size"], conifg["data"]["test_data_size"]
    
    train_max, valid_max, test_max = split_counts
    if frame_count <= train_max:
        return "train"
    elif frame_count <= valid_max:
        return "valid"
    elif frame_count <= test_max:
        return "test"
    else:
        return None

def predict_objects(model, frame):
    return model.predict(frame, verbose=False)[0]

def save_frame_and_labels(frame, results, img_name, txt_name, image_dir, label_dir):
    cv2.imwrite(os.path.join(image_dir, img_name), frame)
    with open(os.path.join(label_dir, txt_name), "w") as f:
        for box in results.boxes:
            cls = int(box.cls)
            cx, cy, w, h = box.xywhn[0].tolist()
            f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

def collect_and_label_frames(model_path="./yolo11n.pt", max_frames=1000, camera_id=0):
    now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    save_root = f"./auto_train_yolo/{now}_datasets"
    folders = init_dataset_folders(save_root)

    model = YOLO(model_path)
    id2name = model.names  # 동적 클래스 이름 맵

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError("카메라 열기에 실패했습니다.")

    frame_count = 0
    detected_class_ids = []

    while frame_count < max_frames:
        success, frame = cap.read()
        if not success:
            print("⚠️ 프레임 읽기 실패")
            break

        frame_count += 1
        split = get_split_type(frame_count)
        if split is None:
            print("✅ 최대 프레임 도달")
            break

        img_name = f"frame_{frame_count:05d}.jpg"
        txt_name = f"frame_{frame_count:05d}.txt"
        image_dir = os.path.join(folders[split], "images")
        label_dir = os.path.join(folders[split], "labels")

        results = predict_objects(model, frame)
        if results.boxes:
            for box in results.boxes:
                detected_class_ids.append(int(box.cls))
            save_frame_and_labels(frame, results, img_name, txt_name, image_dir, label_dir)
            print(f"[{split}] {img_name} 저장 완료")
        else:
            print(f"[{split}] {img_name}: 탐지 없음, 라벨 미생성")

    cap.release()
    cv2.destroyAllWindows()

    # 중복 제거 + 정렬
    unique_class_ids = sorted(set(detected_class_ids))

    return save_root, unique_class_ids, id2name
