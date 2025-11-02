# app.py
# Brain MRI — Alzheimer’s Stage Classifier (EffNetV2-B0, 300x300)
# It will AUTO-DISCOVER:
#   • a .keras or .h5 model anywhere in the repo (prefers brain_effv2b0_infer.keras)
#   • or a SavedModel folder (containing saved_model.pb)
#   • labels.json anywhere in the repo
# Files next to app.py will be preferred if present.

import os, io, json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as v2_preproc

# Make TF quieter on Cloud
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ---------------- constants ----------------
APP_DIR = Path(__file__).parent.resolve()
IMG_SZ = 300

st.set_page_config(page_title="Brain MRI Classifier", layout="wide")
st.title("🧠 Brain MRI — Alzheimer’s Stage Classifier")

# ---------------- search helpers ----------------
def find_labels_json(start: Path) -> Path:
    # Prefer local labels.json
    if (start / "labels.json").exists():
        return start / "labels.json"
    # Else search the repo (first match)
    for p in start.rglob("labels.json"):
        return p
    raise FileNotFoundError("labels.json not found anywhere in the repository.")

def find_model(start: Path) -> Tuple[str, Path]:
    """
    Returns (kind, path) where kind in {"keras","h5","savedmodel"}.
    Preference order:
      1) ./brain_effv2b0_infer.keras
      2) any *.keras
      3) any *.h5
      4) any folder containing saved_model.pb
    """
    # 1) preferred .keras in root
    pref = start / "brain_effv2b0_infer.keras"
    if pref.exists():
        return "keras", pref

    # 2) any .keras (pick largest to avoid tiny placeholders)
    keras_candidates = sorted(start.rglob("*.keras"),
                              key=lambda p: p.stat().st_size if p.exists() else 0,
                              reverse=True)
    if keras_candidates:
        return "keras", keras_candidates[0]

    # 3) any .h5
    h5_candidates = sorted(start.rglob("*.h5"),
                           key=lambda p: p.stat().st_size if p.exists() else 0,
                           reverse=True)
    if h5_candidates:
        return "h5", h5_candidates[0]

    # 4) SavedModel folder
    for pb in start.rglob("saved_model.pb"):
        return "savedmodel", pb.parent

    raise FileNotFoundError(
        "No model file found. Add a .keras or .h5 model, or a SavedModel folder (saved_model.pb)."
    )

# ---------------- model / labels loaders ----------------
def load_labels(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        max_idx = max(int(k) for k in data.keys())
        return [data[str(i)] for i in range(max_idx + 1)]
    if isinstance(data, list):
        return data
    raise ValueError("labels.json must be a dict or a list")

@st.cache_resource(show_spinner=True)
def cached_load_everything(start: Path):
    # Find assets
    labels_path = find_labels_json(start)
    model_kind, model_path = find_model(start)

    # Load labels
    class_names = load_labels(labels_path)

    # Load model
    if model_kind in ("keras", "h5"):
        # Your training used a Lambda registered as "custom>effv2_preproc".
        # We also register plain "preprocess_input" for safety.
        custom = {"custom>effv2_preproc": v2_preproc, "preprocess_input": v2_preproc}
        try:
            model = load_model(str(model_path), custom_objects=custom)
        except TypeError:
            # Some environments reject custom_objects when not present; try plain
            model = load_model(str(model_path))
        is_keras_graph = True  # inspectable; Grad-CAM enabled
    else:
        infer = tf.saved_model.load(str(model_path))
        serving = infer.signatures["serving_default"]

        class Wrap:
            def predict(self, x, verbose=0):
                out = serving(tf.constant(x))
                return next(iter(out.values())).numpy()
            @property
            def layers(self):  # no graph for Grad-CAM
                return []

        model = Wrap()
        is_keras_graph = False

    meta = {
        "labels_path": str(labels_path.relative_to(start)),
        "model_kind": model_kind,
        "model_path": str(model_path.relative_to(start)),
    }
    return model, class_names, is_keras_graph, meta

# ---------------- inference utils ----------------
def preprocess_pil(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((IMG_SZ, IMG_SZ))
    x = np.array(img, dtype=np.float32)[None, ...]
    return v2_preproc(x)

def predict_one(model, img: Image.Image) -> np.ndarray:
    x = preprocess_pil(img)
    return model.predict(x, verbose=0)  # (1, C)

def get_last_conv_name(keras_model) -> Optional[str]:
    last = None
    for lyr in getattr(keras_model, "layers", []):
        if isinstance(lyr, tf.keras.layers.Conv2D):
            last = lyr.name
    return last

def grad_cam(keras_model, img: Image.Image, alpha=0.40):
    # Only for inspectable Keras models
    if not hasattr(keras_model, "layers") or not keras_model.layers:
        return None
    layer_name = get_last_conv_name(keras_model)
    if layer_name is None:
        return None

    x = preprocess_pil(img)
    grad_model = tf.keras.Model([keras_model.inputs],
                                [keras_model.get_layer(layer_name).output, keras_model.output])
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x)
        class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]
    grads = tape.gradient(loss, conv_out)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = tf.reduce_sum(tf.multiply(weights, conv_out[0]), axis=-1)
    cam = tf.maximum(cam, 0) / (tf.reduce_max(cam) + 1e-8)
    cam = tf.image.resize(cam[..., None], (IMG_SZ, IMG_SZ)).numpy().squeeze()

    base = np.asarray(img.convert("RGB").resize((IMG_SZ, IMG_SZ)), dtype=np.float32) / 255.0
    heat = plt.cm.jet(cam)[..., :3]
    overlay = np.clip((1 - alpha) * base + alpha * heat, 0, 1)
    return (overlay * 255).astype(np.uint8)

def pil_to_png_bytes(pil_img: Image.Image) -> bytes:
    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()

# ---------------- load assets & show meta ----------------
try:
    model, class_names, is_keras, meta = cached_load_everything(APP_DIR)
    with st.expander("Loaded assets (auto-discovered)", expanded=False):
        st.code(json.dumps(meta, indent=2))
    st.success(f"Model loaded: {meta['model_kind']} → {meta['model_path']}")
except Exception as e:
    st.error(f"Model/labels discovery or load error: {e}")
    st.stop()

# Best-effort sanity: output size vs labels length
try:
    dummy = v2_preproc(np.zeros((1, IMG_SZ, IMG_SZ, 3), dtype=np.float32))
    out = model.predict(dummy, verbose=0)
    if out.shape[-1] != len(class_names):
        st.warning(f"Model outputs {out.shape[-1]} classes but labels.json has {len(class_names)} entries.")
except Exception:
    pass

st.caption("Classes: " + " | ".join(class_names))
st.markdown("---")

# ---------------- UI ----------------
show_cam = st.checkbox("Show Grad-CAM (only for .keras/.h5 models)", value=True)
uploaded = st.file_uploader("Upload an MRI image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    # Robust read: bytes → PIL
    try:
        img = Image.open(io.BytesIO(uploaded.getvalue())).convert("RGB")
    except Exception as e:
        st.error(f"Could not open image: {e}")
        st.stop()

    st.image(pil_to_png_bytes(img), caption="Input", use_container_width=True)

    # Predict
    try:
        probs = predict_one(model, img)[0]
    except Exception as e:
        st.error(f"Inference failed: {e}")
        st.stop()

    idx = int(np.argmax(probs))
    st.subheader(f"Predicted Stage: **{class_names[idx]}** (confidence {probs[idx]:.3f})")

    # Probability bars
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(range(len(class_names)), probs)
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Grad-CAM
    if show_cam and is_keras:
        cam = grad_cam(model, img)
        if cam is not None:
            st.image(pil_to_png_bytes(Image.fromarray(cam)),
                     caption="Grad-CAM overlay", use_container_width=True)
        else:
            st.caption("Grad-CAM not available for this model.")
    elif show_cam and not is_keras:
        st.caption("Grad-CAM is only available for .keras/.h5 Keras models.")
