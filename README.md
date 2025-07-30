## 📖 Diagrams 사이트
[Diagrams](https://app.diagrams.net)<br>

## 🚩 Diagrams 구성
<p align="center">
  <img src="https://github.com/user-attachments/assets/9ce66cb2-847d-4bb1-aede-53348782682e" width="1000">
</p>

## 📂 폴더 및 파일 구조
```
auto_train_yolo/
├── main.py                      # 파이프라인 실행 진입점
├── utils/
│   ├── config.yaml              # 파이프라인 설정 파일 (학습 파라미터, 카메라 등)
│   └── config_loader.py         # YAML 설정 로더
├── yolo_pipeline.py             # 전체 파이프라인 관리 (수집→학습→평가)
├── yolo_create_data_yaml.py     # 탐지 클래스 기반 data.yaml 생성
├── yolo_detection.py            # YOLO 모델 로드 및 탐지 함수
├── yolo_eval.py                 # 학습된 모델 평가 기능
├── yolo_preprocessing.py        # 영상 수집 및 라벨링, 데이터셋 폴더 생성
├── yolo_postprocess.py          # 학습 후 결과 처리 (모델 교체/데이터 삭제)
├── yolo_train.py                # YOLO 모델 재학습 기능
└── (자동 생성 폴더)              # timestamp 기반 데이터셋 및 학습 결과 저장 폴더
```

## ✅ 실행
```
python3 auto_train_yolo/main.py
```

## 🚀 출력 로그
[TEST TRAIN]<br>
Datasets : Train 7장, Valid 2장, Test 1장(7 : 2 : 1)<br>
EPOCHS : 1<br>
```
[train] frame_00001.jpg 저장 완료
[train] frame_00002.jpg 저장 완료
[train] frame_00003.jpg 저장 완료
[train] frame_00004.jpg 저장 완료
[train] frame_00005.jpg 저장 완료
[train] frame_00006.jpg 저장 완료
[train] frame_00007.jpg 저장 완료
[valid] frame_00008.jpg 저장 완료
[valid] frame_00009.jpg 저장 완료
[test] frame_00010.jpg 저장 완료
✅ 최대 프레임 도달
✅ data.yaml 생성 완료: ./auto_train_yolo/2025_07_30_15_56_59_datasets/data.yaml
 - 클래스 수: 1
 - 클래스명: ['person']
New https://pypi.org/project/ultralytics/8.3.170 available 😃 Update with 'pip install -U ultralytics'
Ultralytics 8.3.161 🚀 Python-3.9.23 torch-2.7.1 CPU (Apple M1)
engine/trainer: agnostic_nms=False, amp=True, augment=False, auto_augment=randaugment, batch=8, bgr=0.0, box=7.5, cache=False, cfg=None, classes=None, close_mosaic=10, cls=0.5, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=./auto_train_yolo/2025_07_30_15_56_59_datasets/data.yaml, degrees=0.0, deterministic=True, device=cpu, dfl=1.5, dnn=False, dropout=0.0, dynamic=False, embed=None, epochs=1, erasing=0.4, exist_ok=True, fliplr=0.5, flipud=0.0, format=torchscript, fraction=1.0, freeze=None, half=False, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=32, int8=False, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.01, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=yolo11n.pt, momentum=0.937, mosaic=1.0, multi_scale=False, name=retrained, nbs=64, nms=False, opset=None, optimize=False, optimizer=auto, overlap_mask=True, patience=100, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=models/retrained, rect=False, resume=False, retina_masks=False, save=True, save_conf=False, save_crop=False, save_dir=models/retrained/retrained, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.5, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=botsort.yaml, translate=0.1, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=8, workspace=None
Overriding model.yaml nc=80 with nc=1

                   from  n    params  module                                       arguments                     
  0                  -1  1       464  ultralytics.nn.modules.conv.Conv             [3, 16, 3, 2]                 
  1                  -1  1      4672  ultralytics.nn.modules.conv.Conv             [16, 32, 3, 2]                
  2                  -1  1      6640  ultralytics.nn.modules.block.C3k2            [32, 64, 1, False, 0.25]      
  3                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]                
  4                  -1  1     26080  ultralytics.nn.modules.block.C3k2            [64, 128, 1, False, 0.25]     
  5                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              
  6                  -1  1     87040  ultralytics.nn.modules.block.C3k2            [128, 128, 1, True]           
  7                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]              
  8                  -1  1    346112  ultralytics.nn.modules.block.C3k2            [256, 256, 1, True]           
  9                  -1  1    164608  ultralytics.nn.modules.block.SPPF            [256, 256, 5]                 
 10                  -1  1    249728  ultralytics.nn.modules.block.C2PSA           [256, 256, 1]                 
 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 13                  -1  1    111296  ultralytics.nn.modules.block.C3k2            [384, 128, 1, False]          
 14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 16                  -1  1     32096  ultralytics.nn.modules.block.C3k2            [256, 64, 1, False]           
 17                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]                
 18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 19                  -1  1     86720  ultralytics.nn.modules.block.C3k2            [192, 128, 1, False]          
 20                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              
 21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 22                  -1  1    378880  ultralytics.nn.modules.block.C3k2            [384, 256, 1, True]           
 23        [16, 19, 22]  1    430867  ultralytics.nn.modules.head.Detect           [1, [64, 128, 256]]           
YOLO11n summary: 181 layers, 2,590,035 parameters, 2,590,019 gradients, 6.4 GFLOPs

Transferred 448/499 items from pretrained weights
Freezing layer 'model.23.dfl.conv.weight'
train: Fast image access ✅ (ping: 0.0±0.0 ms, read: 824.4±159.9 MB/s, size: 238.2 KB)
train: Scanning /Users/jini/Downloads/jini/st_coding/auto_train_yolo/2025_07_30_15_56_59_datasets/train/labels... 7 images, 0 backgrounds, 0 corrupt: 1
train: New cache created: /Users/jini/Downloads/jini/st_coding/auto_train_yolo/2025_07_30_15_56_59_datasets/train/labels.cache
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 728.0±37.2 MB/s, size: 241.0 KB)
val: Scanning /Users/jini/Downloads/jini/st_coding/auto_train_yolo/2025_07_30_15_56_59_datasets/valid/labels... 2 images, 0 backgrounds, 0 corrupt: 100
val: New cache created: /Users/jini/Downloads/jini/st_coding/auto_train_yolo/2025_07_30_15_56_59_datasets/valid/labels.cache
Plotting labels to models/retrained/retrained/labels.jpg... 
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: AdamW(lr=0.002, momentum=0.9) with parameter groups 81 weight(decay=0.0), 88 weight(decay=0.0005), 87 bias(decay=0.0)
Image sizes 32 train, 32 val
Using 0 dataloader workers
Logging results to models/retrained/retrained
Starting training for 1 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        1/1         0G       3.45      3.639       1.44         17         32: 100%|██████████| 1/1 [00:00<00:00,  2.65it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00, 21.75it/s]
                   all          2          2     0.0714          1      0.995       0.31

1 epochs completed in 0.001 hours.
Optimizer stripped from models/retrained/retrained/weights/last.pt, 5.4MB
Optimizer stripped from models/retrained/retrained/weights/best.pt, 5.4MB

Validating models/retrained/retrained/weights/best.pt...
Ultralytics 8.3.161 🚀 Python-3.9.23 torch-2.7.1 CPU (Apple M1)
YOLO11n summary (fused): 100 layers, 2,582,347 parameters, 0 gradients, 6.3 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00, 21.00it/s]
                   all          2          2     0.0714          1      0.995       0.31
Speed: 0.0ms preprocess, 9.1ms inference, 0.0ms loss, 0.4ms postprocess per image
Results saved to models/retrained/retrained
Ultralytics 8.3.161 🚀 Python-3.9.23 torch-2.7.1 CPU (Apple M1)
YOLO11n summary (fused): 100 layers, 2,582,347 parameters, 0 gradients, 6.3 GFLOPs
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 726.3±41.3 MB/s, size: 241.0 KB)
val: Scanning /Users/jini/Downloads/jini/st_coding/auto_train_yolo/2025_07_30_15_56_59_datasets/valid/labels.cache... 2 images, 0 backgrounds, 0 corrup
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00, 31.71it/s]
                   all          2          2     0.0714          1      0.995       0.31
Speed: 0.0ms preprocess, 6.8ms inference, 0.0ms loss, 0.3ms postprocess per image
Results saved to runs/detect/val
모델 평가 결과 - mAP50: 0.995
===== 1 번째 반복 종료 - mAP50: 0.995 =====
```
