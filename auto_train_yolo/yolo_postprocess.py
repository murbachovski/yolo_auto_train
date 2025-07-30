# auto_train_yolo/yolo_postprocess.py

import shutil
import os

def post_process(accuracy, threshold=0.9, new_model_path=None, pretrained_model_path=None, data_dir=None):
    if accuracy >= threshold and new_model_path and pretrained_model_path:
        shutil.copyfile(new_model_path, pretrained_model_path)
        print(f"✅ 모델 교체 완료: {new_model_path} -> {pretrained_model_path}")
    elif accuracy < threshold and data_dir:
        shutil.rmtree(data_dir)
        print(f"⚠️ accuracy 미달로 데이터 삭제: {data_dir}")
