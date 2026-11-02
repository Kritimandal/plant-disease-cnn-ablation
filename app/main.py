import json
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from models import ARMS, build

RESULTS_DIR = ROOT_DIR / "results"
SPLIT_META = ROOT_DIR / "data" / "splits" / "split_meta.json"

st.set_page_config(
    page_title="Plant Disease Ablation Dashboard",
    layout="wide",
)


@st.cache_resource
def load_runs():
    if not SPLIT_META.exists():
        return {}, []

    classes = json.loads(SPLIT_META.read_text(encoding="utf-8"))["classes"]
    runs = {}
    for run_dir in sorted(RESULTS_DIR.glob("*_seed*")):
        results_path = run_dir / "results.json"
        weights_path = run_dir / "best.weights.h5"
        if not (results_path.exists() and weights_path.exists()):
            continue

        meta = json.loads(results_path.read_text(encoding="utf-8"))
        img_size = meta["protocol"]["img_size"]
        model = build(meta["arm"], input_shape=(img_size, img_size, 3), num_classes=len(classes))
        model.load_weights(weights_path)
        runs[meta["arm"]] = {"model": model, "meta": meta, "img_size": img_size}

    return runs, classes


def preprocess(image, img_size):
    image = image.convert("RGB").resize((img_size, img_size))
    return np.expand_dims(np.array(image).astype("float32"), axis=0)


def format_label(raw_label):
    plant, _, disease = raw_label.partition("___")
    return f"{plant.replace('_', ' ')} - {disease.replace('_', ' ')}"


runs, classes = load_runs()

st.title("Selective Adoption of MobileNetV2 Principles: Ablation Dashboard")
st.markdown(
    "This dashboard presents the results of an ablation study evaluating the selective adoption of "
    "MobileNetV2 architectural principles to develop an efficient custom Convolutional Neural Network (CNN) "
    "for plant disease classification. It compares an unmodified custom baseline, four ablation variants, "
    "and a frozen-feature MobileNetV2 reference. All models were trained and evaluated on a leakage-free, "
    "source-grouped dataset."
)

if not runs:
    st.warning(
        "No trained arms found in `results/`. Please execute the ablation sweep first: "
        "`notebooks/run_ablation_colab.ipynb` on a GPU runtime, or "
        "`python scripts/run_ablation.py` locally."
    )
    st.info(
        "Note: Prior models from the original experiment are deliberately excluded. "
        "They were evaluated on a contaminated validation split containing augmented copies of "
        "training photographs, resulting in data leakage. The scores presented here utilize "
        "a strictly source-grouped repartitioning."
    )
    st.stop()

st.caption(f"Loaded arms: {', '.join(sorted(runs))}")

selected = st.multiselect(
    "Arms to compare",
    options=sorted(runs),
    default=sorted(runs),
    format_func=lambda a: f"{a}: {ARMS[a][0]}" if a in ARMS else a,
)

uploaded_file = st.file_uploader("Upload a plant leaf image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and selected:
    image = Image.open(uploaded_file)

    left, right = st.columns([1, 2])
    with left:
        st.markdown("### Uploaded image")
        st.image(image, use_container_width=True)

    with right:
        st.markdown("### Predictions")
        with st.spinner("Running every selected arm..."):
            rows = []
            for arm in selected:
                run = runs[arm]
                probs = run["model"].predict(preprocess(image, run["img_size"]), verbose=0)[0]
                idx = int(np.argmax(probs))
                rows.append({
                    "Arm": arm,
                    "Prediction": format_label(classes[idx]),
                    "Confidence": f"{probs[idx] * 100:.1f}%",
                    "Test macro-F1": f"{run['meta']['test']['macro_f1'] * 100:.1f}%",
                })
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.caption(
            "Confidence indicates the model's certainty on this specific image. 'Test macro-F1' "
            "represents its validated classification performance on the held-out test split. "
            "A high confidence score does not guarantee predictive accuracy."
        )
elif not selected:
    st.info("Select at least one arm.")
else:
    st.info("Awaiting image upload.")

st.markdown("---")
st.markdown("### Measured results")

table = []
for arm in sorted(runs):
    meta = runs[arm]["meta"]
    test, eff, train = meta["test"], meta["efficiency"], meta["training"]
    table.append({
        "Arm": arm,
        "Change": ARMS[arm][0] if arm in ARMS else "",
        "Accuracy": f"{test['accuracy'] * 100:.1f}%",
        "Macro F1": f"{test['macro_f1'] * 100:.1f}%",
        "Parameters": f"{eff['total_params']:,}",
        "Size (MB)": eff["model_size_mb"],
        "CPU latency (ms)": eff["latency_ms_median"],
        "Best epoch": train["best_epoch"],
    })

st.dataframe(table, use_container_width=True, hide_index=True)
st.caption(
    "Performance metrics (Accuracy, Macro-F1) were evaluated on the held-out test split of 9,166 "
    "independent, non-augmented photographs, which no model observed during training or selection. "
    "Latency reflects median single-image CPU inference time to simulate field deployment constraints."
)

figure = ROOT_DIR / "output" / "ablation_tradeoff.png"
if figure.exists():
    st.markdown("### Accuracy against cost")
    st.image(str(figure), use_container_width=True)