"""ResNet18 전이학습으로 NEU 6종 결함 분류 모델을 학습합니다 (10번 노트북 확장판).

- ImageNet 사전학습 가중치에서 시작해 layer4와 마지막 fc만 미세조정합니다.
- 뒤집기·90도 회전 증강을 씁니다. 철강 표면 결함은 방향이 바뀌어도 종류가 같습니다.
- val 손실이 가장 낮은 epoch의 가중치를 저장하고, test는 마지막에 한 번만 평가합니다.

실행: python 02_train_classifier.py
"""
import copy
import json
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from common import APP_DIR, CLASSES, CLS_IMAGE_SIZE, MODELS, RESULTS, build_classifier, to_tensor

torch.manual_seed(42)
np.random.seed(42)
DATA = APP_DIR.parent / "data"
EPOCHS = 15

manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
neu = pd.DataFrame([r for r in manifest if r["dataset"] == "neu"])
class_to_id = {name: i for i, name in enumerate(CLASSES)}


def load_split(split):
    table = neu[neu.split == split].reset_index(drop=True)
    images = []
    for path in table.path:
        with Image.open(DATA / path) as image:
            rgb = np.array(image.convert("RGB").resize((CLS_IMAGE_SIZE, CLS_IMAGE_SIZE), Image.Resampling.BILINEAR))
        images.append(to_tensor(rgb))
    return table, torch.cat(images), torch.tensor(table.kind.map(class_to_id).to_numpy())


def augment(batch):
    """배치 전체에 무작위 좌우·상하 뒤집기와 90도 회전을 적용합니다."""
    if torch.rand(1) < 0.5:
        batch = batch.flip(3)
    if torch.rand(1) < 0.5:
        batch = batch.flip(2)
    return torch.rot90(batch, int(torch.randint(0, 4, (1,))), dims=(2, 3))


train, x_train, y_train = load_split("train")
validation, x_val, y_val = load_split("val")
test, x_test, y_test = load_split("test")
print("train/val/test:", len(train), len(validation), len(test))

network = build_classifier()
pretrained = torch.load(APP_DIR / "work/resnet18-f37072fd.pth", map_location="cpu", weights_only=True)
pretrained = {k: v for k, v in pretrained.items() if not k.startswith("fc.")}
network.load_state_dict(pretrained, strict=False)
for name, parameter in network.named_parameters():
    parameter.requires_grad = name.startswith(("layer4", "fc"))

optimizer = torch.optim.Adam([
    {"params": network.layer4.parameters(), "lr": 1e-4},
    {"params": network.fc.parameters(), "lr": 1e-3},
])
criterion = nn.CrossEntropyLoss()

history = []
best_loss, best_acc, best_state = float("inf"), 0.0, None
started = time.perf_counter()
for epoch in range(1, EPOCHS + 1):
    network.train()
    # 앞쪽 동결 층의 BatchNorm 통계는 ImageNet 값으로 유지합니다.
    for module in [network.bn1, network.layer1, network.layer2, network.layer3]:
        module.eval()
    order = torch.randperm(len(x_train))
    total = 0.0
    for start in range(0, len(order), 16):
        index = order[start:start + 16]
        optimizer.zero_grad()
        loss = criterion(network(augment(x_train[index])), y_train[index])
        loss.backward()
        optimizer.step()
        total += loss.item() * len(index)
    network.eval()
    with torch.no_grad():
        val_logits = network(x_val)
    val_loss = criterion(val_logits, y_val).item()
    val_acc = (val_logits.argmax(1) == y_val).float().mean().item()
    history.append(dict(epoch=epoch, train_loss=total / len(x_train), val_loss=val_loss, val_accuracy=val_acc))
    print(f"epoch {epoch:02d} train_loss={total / len(x_train):.4f} val_loss={val_loss:.4f} val_acc={val_acc:.3f}")
    # val이 36장뿐이라 정확도는 금방 100%에서 동점이 됩니다. 더 세밀한 val 손실로 고릅니다.
    if val_loss < best_loss:
        best_loss, best_acc, best_state = val_loss, val_acc, copy.deepcopy(network.state_dict())
train_seconds = time.perf_counter() - started

network.load_state_dict(best_state)
network.eval()
with torch.no_grad():
    probabilities = torch.cat([network(x_test[i:i + 16]) for i in range(0, len(x_test), 16)]).softmax(1).numpy()
predictions = probabilities.argmax(1)
actual = y_test.numpy()

MODELS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)
# float16으로 저장해 GitHub 웹 업로드 한도(25MB)보다 작게 만듭니다. 불러올 때 float32로 자동 변환됩니다.
torch.save({k: v.half() if v.is_floating_point() else v for k, v in network.state_dict().items()},
           MODELS / "classifier.pt")
pd.DataFrame(history).to_csv(RESULTS / "classifier_history.csv", index=False)
result = test[["path", "kind"]].copy()
result["prediction"] = [CLASSES[i] for i in predictions]
result["score"] = probabilities.max(1)
result.to_csv(RESULTS / "classifier_test_predictions.csv", index=False, encoding="utf-8-sig")
metrics = {
    "model": "ResNet18 전이학습 (layer4+fc 미세조정)",
    "image_size": CLS_IMAGE_SIZE,
    "epochs": EPOCHS,
    "best_val_accuracy": best_acc,
    "test_accuracy": float(accuracy_score(actual, predictions)),
    "test_macro_f1": float(f1_score(actual, predictions, average="macro")),
    "train_seconds": train_seconds,
    "report": classification_report(actual, predictions, labels=list(range(len(CLASSES))),
                                    target_names=CLASSES, output_dict=True, zero_division=0),
    "confusion_matrix": confusion_matrix(actual, predictions, labels=list(range(len(CLASSES)))).tolist(),
}
(RESULTS / "classifier_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
print(classification_report(actual, predictions, target_names=CLASSES, zero_division=0))
print(f"test accuracy={metrics['test_accuracy']:.3f} macro F1={metrics['test_macro_f1']:.3f} "
      f"학습 {train_seconds:.0f}초 -> {MODELS / 'classifier.pt'}")
