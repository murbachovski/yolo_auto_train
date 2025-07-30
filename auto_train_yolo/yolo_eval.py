# auto_train_yolo/yolo_eval.py

def evaluate_model(model, data_yaml_path):
    results = model.val(data=data_yaml_path)
    if hasattr(results.box, "map50") and results.box.map50 is not None:
        mAP50 = results.box.map50
    elif hasattr(results.box, "map") and results.box.map is not None:
        if isinstance(results.box.map, (list, tuple)):
            mAP50 = results.box.map[0]
        else:
            mAP50 = results.box.map
    else:
        mAP50 = None

    print(f"모델 평가 결과 - mAP50: {mAP50}")
    return mAP50
