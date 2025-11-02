# app.py
# Brain MRI — Alzheimer’s Stage Classifier (EffNetV2-B0, 300x300)
# Place these next to this file:
#   - brain_effv2b0_infer.keras   (preferred)  OR  brain_savedmodel/ (fallback folder with saved_model.pb)
#   - labels.json  (index -> class name mapping)

import os, io, json
from pathlib import Path

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as v2_preproc

# ---- quieter TF logs on Cloud ----
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ---------- constants ----------
APP_DIR = Path(__file__).parent.resolve()
KERAS_PATH = APP_DIR / "brain_effv2b0_infer.keras"
LABELS_PATH = APP_DIR / "labels.json"
SAVEDMODEL_DIR = APP_DIR / "brain_savedmodel"   # contains saved_model.pb if used
IMG_SZ = 300

st.set_page_config(page_title="Brain MRI Classifier", layout="wide")
st.title("🧠 Brain MRI — Alzheimer’s Stage Classifier")

# ---------- helpers ----------
def load_labels(path: Path):
    """Load labels.json and normalize to a simple list [idx->name]."""
    data = json.loads(path.read_text(encoding="utf-8"))
    # Accept either {"0":"No","1":"Mild",...} or ["No","Mild",...]
    if isinstance(data, dict):
        # make sure keys are sorted numerically
        max_idx = max(int(k) for k in data.keys())
        lst = []
        for i in range(max_idx + 1):
            key = str(i)
            if key not in data:
                raise ValueError(f"labels.json missing index '{i}'")
            lst.append(data[key])
        return lst
    elif isinstance(data, list):
        return data
    else:
        raise ValueError("labels.json must be a dict or list.")

def preprocess_pil(img: Image.Image) -> np.ndarray:
    """PIL.Image -> preprocessed tensor batch (1, H, W, 3)."""
    img = img.convert("RGB").resize((IMG_SZ, IMG_SZ))
    x = np.array(img, dtype=np.float32)[None, ...]
    return v2_preproc(x)

def predict_one(model, img: Image.Image) -> np.ndarray:
    x = preprocess_pil(img)
    return model.predict(x, verbose=0)  # (1, C)

def get_last_conv_name(keras_model):
    last = None
    for lyr in getattr(keras_model, "layers", []):
        if isinstance(lyr, tf.keras.layers.Conv2D):
            last = lyr.name
    return last

def grad_cam(keras_model, img: Image.Image, alpha=0.40):
    """Compute Grad-CAM overlay for a Keras Functional/Sequential model."""
    # Only possible if we have an inspectable Keras model
    if not hasattr(keras_model, "layers") or not keras_model.layers:
        return None
    layer_name = get_last_conv_name(keras_model)
    if layer_name is None:
        return None

    x = preprocess_pil(img)
    grad_model = tf.keras.Model(
        [keras_model.inputs],
        [keras_model.get_layer(layer_name).output, keras_model.output]
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x)
        class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]
    grads = tape.gradient(loss, conv_out)[0]            # (H, W, C)
    weights = tf.reduce_mean(grads, axis=(0, 1))        # (C,)
    cam = tf.reduce_sum(tf.multiply(weights, conv_out[0]), axis=-1)  # (H, W)
    cam = tf.maximum(cam, 0) / (tf.reduce_max(cam) + 1e-8)
    cam = tf.image.resize(cam[..., None], (IMG_SZ, IMG_SZ)).numpy().squeeze()

    base = np.array(img.convert("RGB").resize((IMG_SZ, IMG_SZ)), np.float32) / 255.0
    heat = plt.cm.jet(cam)[..., :3]
    overlay = np.clip((1 - alpha) * base + alpha * heat, 0, 1)
    return (overlay * 255).astype(np.uint8)

# ---------- cached loaders (prevents reloading every rerun) ----------
@st.cache_resource(show_spinner=True)
def cached_load_labels(labels_path: Path):
    return load_labels(labels_path)

@st.cache_resource(show_spinner=True)
def cached_load_model(keras_path: Path, savedmodel_dir: Path):
    """
    Try to load .keras first (with custom Lambda name "custom>effv2_preproc"),
    else fall back to SavedModel directory.
    Returns: (model_like, is_keras_bool)
    """
    if keras_path.exists():
        # Your .keras was saved with a Lambda registered as "custom>effv2_preproc"
        m = load_model(str(keras_path), custom_objects={"custom>effv2_preproc": v2_preproc})
        return m, True

    sm_pb = savedmodel_dir / "saved_model.pb"
    if sm_pb.exists():
        infer = tf.saved_model.load(str(savedmodel_dir))
        serving = infer.signatures["serving_default"]

        class Wrap:
            def predict(self, x, verbose=0):
                out = serving(tf.constant(x))
                return next(iter(out.values())).numpy()
            @property
            def layers(self):
                # No layer graph available; Grad-CAM disabled
                return []

        return Wrap(), False

    raise FileNotFoundError(
        "No model found. Put 'brain_effv2b0_infer.keras' or 'brain_savedmodel/' next to app."
    )

# ---------- load assets ----------
try:
    class_names = cached_load_labels(LABELS_PATH)
except Exception as e:
    st.error(f"Could not read labels.json: {e}")
    st.stop()

try:
    model, is_keras = cached_load_model(KERAS_PATH, SAVEDMODEL_DIR)
    st.success("Model loaded." + (" (Keras)" if is_keras else " (SavedModel)"))
except Exception as e:
    st.error(f"Model load error: {e}")
    st.stop()

# quick sanity for class count vs model output (best-effort)
try:
    dummy = np.zeros((1, IMG_SZ, IMG_SZ, 3), dtype=np.float32)
    dummy = v2_preproc(dummy)
    out = model.predict(dummy, verbose=0)
    if out.shape[-1] != len(class_names):
        st.warning(
            f"Model outputs {out.shape[-1]} classes but labels.json has {len(class_names)} "
            f"entries. Ensure they match."
        )
except Exception:
    # If SavedModel wrapper without eager predict on dummy, just skip the check.
    pass

st.caption("Classes: " + " | ".join(class_names))
st.markdown("---")

# ---------- UI: upload & predict (Cloud-robust) ----------
show_cam = st.checkbox("Show Grad-CAM (only for .keras models)", value=True)
uploaded = st.file_uploader("Upload an MRI image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    # Robustly read the file bytes → PIL → numpy for st.image
    try:
        data = uploaded.getvalue()
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        st.error(f"Could not open image: {e}")
        st.stop()

    # Show image using NumPy array to avoid object-type quirks on Cloud
    st.image(np.array(img), caption="Input", use_container_width=True)

    # Predict
    try:
        probs = predict_one(model, img)[0]
    except Exception as e:
        st.error(f"Inference failed: {e}")
        st.stop()

    top = int(np.argmax(probs))
    st.subheader(f"Predicted Stage: **{class_names[top]}** (confidence {probs[top]:.3f})")

    # Probability bars
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(range(len(class_names)), probs)
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Grad-CAM only for inspectable Keras models
    if show_cam and is_keras:
        overlay = grad_cam(model, img)
        if overlay is not None:
            st.image(overlay, caption="Grad-CAM overlay", use_container_width=True)
        else:
            st.caption("Grad-CAM could not be generated for this model.")
    elif show_cam and not is_keras:
        st.caption("Grad-CAM is unavailable for SavedModel inference wrapper.")
