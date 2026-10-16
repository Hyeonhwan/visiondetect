"""철강 표면 결함 분류·객체탐지 대시보드 (NEU-DET 6종)

실행: streamlit run app.py
"""
import json

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

from common import (CLASS_KO, CLASSES, CLS_IMAGE_SIZE, DET_IMAGE_SIZE, MODELS, RESULTS, SAMPLES,
                    build_classifier, setup_yolo_env, to_tensor)

st.set_page_config(page_title="철강 결함 분류·탐지 대시보드", page_icon="🔍", layout="wide")
torch.set_num_threads(2)

# 클래스별 표시 색 (RGB) — 차트와 박스에 같은 색을 씁니다.
COLORS = {
    "crazing": (230, 57, 70), "inclusion": (247, 140, 0), "patches": (42, 157, 143),
    "pitted_surface": (58, 110, 230), "rolled-in_scale": (131, 56, 236), "scratches": (214, 40, 140),
}
COLOR_SCALE = alt.Scale(domain=CLASSES, range=[f"rgb{c}" for c in COLORS.values()])


# ---------------------------------------------------------------- 불러오기
@st.cache_resource
def load_classifier():
    path = MODELS / "classifier.pt"
    if not path.is_file():
        return None
    network = build_classifier()
    network.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return network.eval()


@st.cache_resource
def load_detector():
    path = MODELS / "detector.pt"
    if not path.is_file():
        return None
    setup_yolo_env()
    from ultralytics import YOLO
    return YOLO(str(path))


@st.cache_data
def load_json(name):
    path = RESULTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


@st.cache_data
def load_csv(name):
    path = RESULTS / name
    return pd.read_csv(path) if path.is_file() else None


@st.cache_data
def sample_labels():
    return json.loads((SAMPLES / "labels.json").read_text(encoding="utf-8"))


def read_image(source):
    with Image.open(source) as image:
        return np.array(image.convert("RGB"))


# ---------------------------------------------------------------- 모델 실행
def classify(network, rgb):
    """6종 확률과 예측 클래스의 Grad-CAM(0~1, 원본 크기)을 돌려줍니다."""
    resized = cv2.resize(rgb, (CLS_IMAGE_SIZE, CLS_IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    captured = {}

    def hook(module, inputs, output):
        captured["activation"] = output
        output.retain_grad()

    handle = network.layer4.register_forward_hook(hook)
    try:
        logits = network(to_tensor(resized))
        probabilities = logits.softmax(1)[0].detach().numpy()
        network.zero_grad()
        logits[0, int(probabilities.argmax())].backward()
    finally:
        handle.remove()
    activation = captured["activation"]
    weights = activation.grad.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activation).sum(dim=1))[0].detach().numpy()
    cam = cv2.resize(cam, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    return probabilities, cam / max(float(cam.max()), 1e-12)


def detect(detector, rgb, conf, iou):
    result = detector.predict(source=cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), imgsz=DET_IMAGE_SIZE,
                              conf=conf, iou=iou, device="cpu", verbose=False)[0]
    rows = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        kind = CLASSES[int(box.cls.item())]
        rows.append(dict(kind=kind, score=float(box.conf.item()), x1=x1, y1=y1, x2=x2, y2=y2))
    return pd.DataFrame(rows, columns=["kind", "score", "x1", "y1", "x2", "y2"])


# ---------------------------------------------------------------- 그리기
def draw_boxes(rgb, boxes, display=480):
    """작은 원본(200px)을 확대한 뒤 박스와 라벨을 그립니다."""
    scale = display / max(rgb.shape[:2])
    canvas = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    for b in boxes:
        color = COLORS[b["kind"]]
        p1 = (int(b["x1"] * scale), int(b["y1"] * scale))
        p2 = (int(b["x2"] * scale), int(b["y2"] * scale))
        cv2.rectangle(canvas, p1, p2, color, 2)
        text = b["kind"] if b.get("score") is None else f"{b['kind']} {b['score']:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(p1[1], th + 6)
        cv2.rectangle(canvas, (p1[0], ty - th - 6), (p1[0] + tw + 6, ty), color, -1)
        cv2.putText(canvas, text, (p1[0] + 3, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)
    return canvas


def overlay_cam(rgb, cam, alpha=0.45):
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return np.uint8((1 - alpha) * rgb + alpha * heat)


def bar_chart(df, x, y, height=260, x_title=None, x_format=None):
    return alt.Chart(df).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X(x, title=x_title, axis=alt.Axis(format=x_format) if x_format else alt.Axis()),
        y=alt.Y(y, sort=CLASSES, title=None),
        color=alt.Color("kind:N", scale=COLOR_SCALE, legend=None),
        tooltip=list(df.columns),
    ).properties(height=height)


def confusion_chart(matrix):
    df = pd.DataFrame(matrix, index=CLASSES, columns=CLASSES).reset_index(names="actual")
    df = df.melt(id_vars="actual", var_name="predicted", value_name="count")
    base = alt.Chart(df).encode(
        x=alt.X("predicted:N", sort=CLASSES, title="예측"),
        y=alt.Y("actual:N", sort=CLASSES, title="정답"),
    )
    rect = base.mark_rect().encode(color=alt.Color("count:Q", scale=alt.Scale(scheme="blues"), legend=None),
                                   tooltip=["actual", "predicted", "count"])
    text = base.mark_text(fontSize=14).encode(
        text="count:Q", color=alt.condition("datum.count > 5", alt.value("white"), alt.value("#333")))
    return (rect + text).properties(height=340)


# ---------------------------------------------------------------- 사이드바: 검사 이미지 선택
classifier = load_classifier()
detector = load_detector()
labels = sample_labels()

with st.sidebar:
    st.header("🔍 검사 이미지")
    source = st.segmented_control("입력 방식", ["예시 이미지", "직접 업로드"], default="예시 이미지", required=True)
    truth = None
    if source == "예시 이미지":
        name = st.selectbox("NEU test 이미지", list(labels),
                            format_func=lambda n: f"{n}  ({CLASS_KO[labels[n]['kind']]})")
        image_rgb = read_image(SAMPLES / name)
        truth = labels[name]
    else:
        uploaded = st.file_uploader("철강 표면 사진 (jpg·png·bmp)", type=["jpg", "jpeg", "png", "bmp"])
        image_rgb = read_image(uploaded) if uploaded else None
        name = uploaded.name if uploaded else None
    if image_rgb is not None:
        st.image(image_rgb, caption=f"{name} · {image_rgb.shape[1]}×{image_rgb.shape[0]}", width="stretch")
    st.divider()
    st.caption("모델 상태")
    st.write(("✅" if classifier else "❌") + " 분류: ResNet18")
    st.write(("✅" if detector else "❌") + " 탐지: YOLOv8n")
    st.caption("학습 데이터: NEU 철강 표면 결함 6종 (정상 사진 없음)")

# ---------------------------------------------------------------- 본문
st.title("철강 표면 결함 분류 · 객체탐지 대시보드")
st.caption("NEU 철강 표면 결함 데이터로 학습한 **분류 모델(ResNet18)**과 **객체탐지 모델(YOLOv8n)**로 "
           "결함 종류와 위치를 확인합니다.")

tab_data, tab_cls, tab_det, tab_perf = st.tabs(["📊 데이터 소개", "🏷️ 결함 분류", "🎯 객체 탐지", "📈 모델 성능"])

# ---- 데이터 소개
with tab_data:
    summary = load_json("data_summary.json")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("전체 이미지", f"{summary['total']}장", border=True)
    c2.metric("결함 종류", f"{len(CLASSES)}종", border=True)
    c3.metric("정답 박스", f"{sum(summary['box_counts'].values())}개", border=True)
    c4.metric("이미지 크기", "×".join(map(str, summary["image_size"])), border=True)

    left, right = st.columns(2)
    with left:
        st.subheader("종류별 · 분할별 이미지 수")
        rows = [dict(kind=k, split=s, count=n) for k, d in summary["class_split_counts"].items() for s, n in d.items()]
        chart = alt.Chart(pd.DataFrame(rows)).mark_bar().encode(
            x=alt.X("count:Q", title="이미지 수"),
            y=alt.Y("kind:N", sort=CLASSES, title=None),
            color=alt.Color("split:N", sort=["train", "val", "test"], title="분할",
                            scale=alt.Scale(domain=["train", "val", "test"], range=["#3a6ee6", "#9ab3f0", "#f59e0b"])),
            order=alt.Order("split_order:Q"),
            tooltip=["kind", "split", "count"],
        ).transform_calculate(split_order="indexof(['train','val','test'], datum.split)").properties(height=260)
        st.altair_chart(chart, width="stretch")
        st.caption(f"분할: train {summary['split_counts']['train']} · val {summary['split_counts']['val']} · "
                   f"test {summary['split_counts']['test']}. 종류마다 같은 수로 나뉘어 있습니다.")
    with right:
        st.subheader("종류별 정답 박스 수")
        boxes = pd.DataFrame([dict(kind=k, boxes=summary["box_counts"].get(k, 0)) for k in CLASSES])
        st.altair_chart(bar_chart(boxes, "boxes:Q", "kind:N", x_title="박스 수"), width="stretch")
        st.caption("한 사진에 결함 영역이 여러 개면 박스도 여러 개입니다 (사진당 1~9개).")

    st.subheader("결함 종류 예시")
    cols = st.columns(len(CLASSES))
    for col, kind in zip(cols, CLASSES):
        first = next(n for n, v in labels.items() if v["kind"] == kind)
        col.image(read_image(SAMPLES / first), caption=f"{kind}\n{CLASS_KO[kind]}", width="stretch")
    st.info("NEU에는 정상(양품) 사진이 없습니다. 두 모델 모두 '정상/불량'이 아니라 **어떤 결함인지**를 판단합니다.")

# ---- 결함 분류
with tab_cls:
    if classifier is None:
        st.warning("models/classifier.pt가 없습니다. `python 02_train_classifier.py`를 먼저 실행하세요.")
    elif image_rgb is None:
        st.info("왼쪽 사이드바에서 이미지를 선택하거나 업로드하세요.")
    else:
        probabilities, cam = classify(classifier, image_rgb)
        top = int(probabilities.argmax())
        left, right = st.columns([1, 1.2])
        with left:
            show_cam = st.toggle("Grad-CAM 겹쳐 보기", value=True,
                                 help="모델이 이 클래스를 고를 때 민감하게 반응한 영역입니다. 결함 위치 정답은 아닙니다.")
            shown = overlay_cam(image_rgb, cam) if show_cam else image_rgb
            st.image(cv2.resize(shown, (420, 420), interpolation=cv2.INTER_NEAREST), width="stretch")
        with right:
            m1, m2 = st.columns(2)
            m1.metric("예측 결함", CLASSES[top], CLASS_KO[CLASSES[top]], delta_color="off", border=True)
            m2.metric("예측 점수", f"{probabilities[top]:.1%}", border=True)
            if truth:
                if truth["kind"] == CLASSES[top]:
                    st.success(f"정답 {truth['kind']} — 맞혔습니다.")
                else:
                    st.error(f"정답 {truth['kind']} — 틀렸습니다.")
            df = pd.DataFrame(dict(kind=CLASSES, 한글=[CLASS_KO[k] for k in CLASSES], probability=probabilities))
            st.altair_chart(bar_chart(df, "probability:Q", "kind:N", x_title="softmax 점수", x_format="%"),
                            width="stretch")
            st.caption("softmax 점수는 6종 중 상대적인 값이며, 보정된 불량 확률이 아닙니다. "
                       "학습에 없던 종류의 사진도 6종 중 하나로 분류됩니다.")

# ---- 객체 탐지
with tab_det:
    if detector is None:
        st.warning("models/detector.pt가 없습니다. `python 03_train_detector.py`를 먼저 실행하세요.")
    elif image_rgb is None:
        st.info("왼쪽 사이드바에서 이미지를 선택하거나 업로드하세요.")
    else:
        s1, s2 = st.columns(2)
        conf = s1.slider("신뢰도 임계값 (conf)", 0.05, 0.95, 0.25, 0.05,
                         help="이 값보다 점수가 낮은 박스는 표시하지 않습니다. 낮추면 놓침은 줄고 오검출은 늘어납니다.")
        iou = s2.slider("중복 제거 IoU (NMS)", 0.1, 0.9, 0.5, 0.05,
                        help="두 박스가 이 비율 이상 겹치면 점수가 낮은 쪽을 지웁니다.")
        found = detect(detector, image_rgb, conf, iou)
        pictures = st.columns(2 if truth else 1)
        with pictures[0]:
            st.markdown(f"**예측** · 결함 후보 {len(found)}개")
            st.image(draw_boxes(image_rgb, found.to_dict("records")), width="stretch")
        if truth:
            with pictures[1]:
                st.markdown(f"**정답** · 박스 {len(truth['boxes'])}개")
                gt = [dict(kind=b["kind"], x1=b["xyxy"][0], y1=b["xyxy"][1], x2=b["xyxy"][2], y2=b["xyxy"][3])
                      for b in truth["boxes"]]
                st.image(draw_boxes(image_rgb, gt), width="stretch")
        if found.empty:
            st.warning("현재 임계값에서 검출된 결함 후보가 없습니다. 결함이 없다는 보장은 아닙니다 — conf를 낮춰 보세요.")
        else:
            table = found.assign(한글=found.kind.map(CLASS_KO)).round({"x1": 0, "y1": 0, "x2": 0, "y2": 0})
            st.dataframe(table[["kind", "한글", "score", "x1", "y1", "x2", "y2"]], hide_index=True,
                         column_config={"score": st.column_config.ProgressColumn("score", min_value=0, max_value=1,
                                                                                format="%.2f")})

# ---- 모델 성능
with tab_perf:
    cls_metrics = load_json("classifier_metrics.json")
    det_metrics = load_json("detector_metrics.json")
    st.caption("모든 수치는 학습·모델 선택에 쓰지 않은 **test 60장**에서 한 번 평가한 값입니다.")
    left, right = st.columns(2)
    with left:
        st.subheader("🏷️ 분류 · ResNet18")
        if cls_metrics:
            a, b, c = st.columns(3)
            a.metric("Accuracy", f"{cls_metrics['test_accuracy']:.1%}", border=True)
            b.metric("Macro F1", f"{cls_metrics['test_macro_f1']:.3f}", border=True)
            c.metric("학습 시간", f"{cls_metrics['train_seconds']:.0f}초", border=True)
            st.markdown("**혼동행렬**")
            st.altair_chart(confusion_chart(cls_metrics["confusion_matrix"]), width="stretch")
            report = pd.DataFrame(cls_metrics["report"]).T.loc[CLASSES, ["precision", "recall", "f1-score"]]
            st.dataframe(report.style.format("{:.2f}").background_gradient(cmap="Blues", vmin=0, vmax=1))
            history = load_csv("classifier_history.csv")
            st.markdown("**학습 곡선**")
            st.line_chart(history.set_index("epoch")[["train_loss", "val_loss"]], height=220)
        else:
            st.warning("분류 결과 파일이 없습니다.")
    with right:
        st.subheader("🎯 객체탐지 · YOLOv8n")
        if det_metrics:
            a, b, c, d = st.columns(4)
            a.metric("mAP50", f"{det_metrics['test_map50']:.3f}", border=True)
            b.metric("mAP50-95", f"{det_metrics['test_map50_95']:.3f}", border=True)
            c.metric("Precision", f"{det_metrics['test_precision']:.3f}", border=True)
            d.metric("Recall", f"{det_metrics['test_recall']:.3f}", border=True)
            st.markdown("**종류별 AP50**")
            ap = pd.DataFrame([dict(kind=k, ap50=det_metrics["per_class_ap50"].get(k, 0.0)) for k in CLASSES])
            st.altair_chart(bar_chart(ap, "ap50:Q", "kind:N", height=340, x_title="AP50"), width="stretch")
            history = load_csv("detector_history.csv")
            st.markdown("**학습 곡선 (val)**")
            curves = history.set_index("epoch")[["metrics/mAP50(B)", "metrics/mAP50-95(B)"]]
            st.line_chart(curves.rename(columns=lambda c: c.split("/")[1].replace("(B)", "")), height=220)
            st.caption(f"학습 {det_metrics['epochs_run']} epoch · {det_metrics['train_seconds'] / 60:.0f}분 (CPU)")
        else:
            st.warning("탐지 결과 파일이 없습니다. `python 03_train_detector.py` 실행 후 새로고침하세요.")

    with st.expander("분류 모델 test 예측 전체 보기"):
        predictions = load_csv("classifier_test_predictions.csv")
        if predictions is not None:
            only_wrong = st.checkbox("틀린 것만 보기")
            view = predictions[predictions.kind != predictions.prediction] if only_wrong else predictions
            st.dataframe(view, hide_index=True)

    st.markdown("""
**두 모델의 차이**
- **분류(ResNet18)**: 사진 한 장 전체에 대해 6종 중 하나를 고릅니다. 결함 위치는 출력하지 않습니다 (Grad-CAM은 참고용).
- **객체탐지(YOLOv8n)**: 결함 영역마다 박스·종류·점수를 출력합니다. 한 사진에 결함이 여러 개여도 각각 찾습니다.
""")
