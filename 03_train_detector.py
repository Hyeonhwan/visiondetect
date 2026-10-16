"""YOLOv8n으로 NEU 결함 위치(박스)와 종류를 학습합니다 (11번 노트북 확장판).

수업에서는 절차 확인용으로 2 epoch만 학습했습니다. 여기서는 박스가 실제로 나오도록 더 길게 학습합니다.
실행: python 03_train_detector.py  (01_prepare_data.py를 먼저 실행)
"""
import json
import shutil
import time
from pathlib import Path

import pandas as pd

from common import APP_DIR, CLASSES, DET_IMAGE_SIZE, MODELS, RESULTS, setup_yolo_env

setup_yolo_env()
from ultralytics import YOLO  # noqa: E402  (환경변수 설정 뒤에 불러옵니다)

EPOCHS = 150
WORK = APP_DIR / "work"
data_yaml = WORK / "neu_yolo/data.yaml"

model = YOLO(str(WORK / "yolov8n.pt"))
started = time.perf_counter()
model.train(data=str(data_yaml), epochs=EPOCHS, imgsz=DET_IMAGE_SIZE, batch=16, device="cpu", workers=0,
            amp=False, plots=False, seed=42, deterministic=True, project=str(WORK / "yolo"),
            name="neu", exist_ok=True, verbose=False, patience=40,
            fliplr=0.5, flipud=0.5, mosaic=1.0, close_mosaic=10)
train_seconds = time.perf_counter() - started
best_path = Path(model.trainer.best)

MODELS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)
shutil.copy2(best_path, MODELS / "detector.pt")
history = pd.read_csv(best_path.parent.parent / "results.csv")
history.columns = [c.strip() for c in history.columns]
history.to_csv(RESULTS / "detector_history.csv", index=False)

# 학습·임계값 선택에 쓰지 않은 test로 최종 평가합니다.
trained = YOLO(str(MODELS / "detector.pt"))
evaluation = trained.val(data=str(data_yaml), split="test", imgsz=DET_IMAGE_SIZE, batch=16, device="cpu",
                         workers=0, plots=False, verbose=False, project=str(WORK / "yolo"), name="neu_test",
                         exist_ok=True)
per_class_ap50 = {CLASSES[int(c)]: float(evaluation.box.ap50[i]) for i, c in enumerate(evaluation.box.ap_class_index)}
metrics = {
    "model": "YOLOv8n (COCO 사전학습 → NEU 미세조정)",
    "image_size": DET_IMAGE_SIZE,
    "epochs_run": int(len(history)),
    "test_map50": float(evaluation.box.map50),
    "test_map50_95": float(evaluation.box.map),
    "test_precision": float(evaluation.box.mp),
    "test_recall": float(evaluation.box.mr),
    "per_class_ap50": per_class_ap50,
    "train_seconds": train_seconds,
}
(RESULTS / "detector_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(metrics, ensure_ascii=False, indent=2))
print("저장:", MODELS / "detector.pt")
