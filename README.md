# 철강 표면 결함 분류 · 객체탐지 대시보드

NEU 철강 표면 결함 데이터(6종)로 **분류 모델(ResNet18 전이학습)**과 **객체탐지 모델(YOLOv8n)**을 학습하고,
Streamlit 대시보드에서 이미지를 넣어 결함 종류와 위치를 확인합니다.

| 결함 | 한글 |
|---|---|
| crazing | 균열(크레이징) |
| inclusion | 개재물 |
| patches | 얼룩(패치) |
| pitted_surface | 점식(피팅) |
| rolled-in_scale | 압입 스케일 |
| scratches | 스크래치 |

## 대시보드 구성

| 탭 | 내용 |
|---|---|
| 📊 데이터 소개 | 종류·분할별 이미지 수, 정답 박스 수, 종류별 예시 |
| 🏷️ 결함 분류 | 6종 점수 막대그래프, Grad-CAM, 예시 이미지면 정답 비교 |
| 🎯 객체 탐지 | conf·IoU 조절, 예측 박스와 정답 박스 비교, 검출 목록 |
| 📈 모델 성능 | test 정확도·F1·혼동행렬, mAP·종류별 AP50, 학습 곡선 |

## 폴더 구조

```
app.py                  Streamlit 대시보드
common.py               학습·앱 공통 설정 (클래스, 모델 구조, 전처리)
01_prepare_data.py      수업 data → 요약·예시 이미지·YOLO 학습 폴더
02_train_classifier.py  ResNet18 전이학습 (layer4+fc 미세조정)
03_train_detector.py    YOLOv8n 미세조정
models/                 classifier.pt, detector.pt, Arial.ttf
results/                데이터 요약, test 성능, 학습 곡선
samples/                대시보드 예시 이미지(test 12장)와 정답 박스
requirements.txt
```

## 실행 방법 (Python 3.13)

```bash
# 학습 (visionproject/data 필요, 이미 학습된 models/가 있으면 생략 가능)
python 01_prepare_data.py
python 02_train_classifier.py
python 03_train_detector.py

# 대시보드
streamlit run app.py
```

## 데이터 분할과 평가 원칙

- train 144 / val 36 / test 60장 (종류별 24 / 6 / 10장)
- 모델 선택은 val로만 하고, test는 최종 평가에 한 번만 사용했습니다.
- NEU에는 정상 사진이 없으므로 정상/불량 판정이 아니라 **결함 종류 판단** 모델입니다.
