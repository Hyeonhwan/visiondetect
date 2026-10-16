"""수업 data 폴더의 NEU 자료로 대시보드용 요약·샘플과 YOLO 학습 폴더를 만듭니다.

실행: python 01_prepare_data.py  (visionproject/data 가 있어야 합니다)
"""
import json
import shutil
from collections import Counter

import yaml
from PIL import Image

from common import APP_DIR, CLASSES, MODELS, RESULTS, SAMPLES

COURSE = APP_DIR.parent
DATA = COURSE / "data"
WORK = APP_DIR / "work"

for folder in (MODELS, RESULTS, SAMPLES, WORK):
    folder.mkdir(exist_ok=True)

manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
neu = [row for row in manifest if row["dataset"] == "neu"]
class_to_id = {name: i for i, name in enumerate(CLASSES)}

# 1) 데이터 요약 (대시보드 '데이터 소개' 탭)
summary = {
    "total": len(neu),
    "split_counts": dict(Counter(r["split"] for r in neu)),
    "class_split_counts": {
        name: dict(Counter(r["split"] for r in neu if r["kind"] == name)) for name in CLASSES
    },
    "box_counts": dict(Counter(a["kind"] for r in neu for a in r["boxes"])),
    "boxes_per_image": dict(Counter(len(r["boxes"]) for r in neu)),
    "image_size": list(Image.open(DATA / neu[0]["path"]).size),
}
(RESULTS / "data_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

# 2) 대시보드 예시 이미지: test에서 클래스별 2장과 정답 박스
labels = {}
for name in CLASSES:
    for row in [r for r in neu if r["split"] == "test" and r["kind"] == name][:2]:
        filename = (DATA / row["path"]).name
        shutil.copy2(DATA / row["path"], SAMPLES / filename)
        labels[filename] = {"kind": row["kind"], "boxes": row["boxes"]}
(SAMPLES / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")

# 3) YOLO 형식 학습 폴더 (11번 노트북과 같은 변환)
yolo_root = WORK / "neu_yolo"
for row in neu:
    image_dir = yolo_root / "images" / row["split"]
    label_dir = yolo_root / "labels" / row["split"]
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    source = DATA / row["path"]
    shutil.copy2(source, image_dir / source.name)
    w, h = Image.open(source).size
    lines = []
    for box in row["boxes"]:
        x1, y1, x2, y2 = box["xyxy"]
        assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h
        lines.append(f"{class_to_id[box['kind']]} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                     f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    (label_dir / f"{source.stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

config = {"path": str(yolo_root.resolve()), "train": "images/train", "val": "images/val",
          "test": "images/test", "names": dict(enumerate(CLASSES))}
(yolo_root / "data.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")

# 4) 수업에서 받은 사전학습 가중치와 글꼴을 앱 폴더로 복사
for name in ("resnet18-f37072fd.pth", "yolov8n.pt", "Arial.ttf"):
    target = WORK / name if name.endswith((".pth", ".pt")) else MODELS / name
    shutil.copy2(COURSE / "models" / name, target)

print("NEU 사진:", summary["total"], summary["split_counts"])
print("예시 이미지:", len(labels), "장 ->", SAMPLES)
print("YOLO 데이터:", yolo_root / "data.yaml")
