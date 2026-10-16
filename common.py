"""학습 스크립트와 대시보드가 함께 쓰는 설정과 함수입니다."""
from pathlib import Path
import os
import shutil

import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18

APP_DIR = Path(__file__).resolve().parent
MODELS = APP_DIR / "models"
RESULTS = APP_DIR / "results"
SAMPLES = APP_DIR / "samples"

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
CLASS_KO = {
    "crazing": "균열(크레이징)",
    "inclusion": "개재물",
    "patches": "얼룩(패치)",
    "pitted_surface": "점식(피팅)",
    "rolled-in_scale": "압입 스케일",
    "scratches": "스크래치",
}

CLS_IMAGE_SIZE = 224
DET_IMAGE_SIZE = 224
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def build_classifier():
    """ResNet18 구조에 6종 결함 분류용 마지막 층을 붙입니다."""
    network = resnet18(weights=None)
    network.fc = nn.Linear(network.fc.in_features, len(CLASSES))
    return network


def to_tensor(rgb):
    """RGB uint8 배열(H,W,3)을 정규화된 (1,3,H,W) 텐서로 바꿉니다. 크기는 미리 맞춰 둡니다."""
    array = rgb.transpose(2, 0, 1).astype(np.float32) / 255
    return (torch.from_numpy(array).unsqueeze(0) - IMAGENET_MEAN) / IMAGENET_STD


def setup_yolo_env():
    """YOLO가 인터넷에서 설정·글꼴을 내려받지 않도록 앱 폴더 안에 준비합니다."""
    config_dir = APP_DIR / ".yolo_config"
    config_dir.mkdir(exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config_dir)
    os.environ["YOLO_OFFLINE"] = "true"
    font = MODELS / "Arial.ttf"
    if font.is_file() and not (config_dir / "Arial.ttf").is_file():
        shutil.copy2(font, config_dir / "Arial.ttf")
