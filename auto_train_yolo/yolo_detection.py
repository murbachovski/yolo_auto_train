# auto_train_yolo/yolo_detection.py

from ultralytics import YOLO

def load_yolo_model(model_path):
    return YOLO(model_path)

def predict_objects(model, frame):
    return model.predict(frame, verbose=False)[0]
