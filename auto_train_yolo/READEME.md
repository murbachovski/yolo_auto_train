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
