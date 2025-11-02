# app.py
# Brain MRI — Alzheimer’s Stage Classifier (EffNetV2-B0, 300x300)
# Expects these files in the SAME folder as this app:
#   - brain_effv2b0_infer.keras   (preferred)  OR  brain_savedmodel/ (fallback)
#   - labels.json

import os, io, json
from pathlib import Path

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as v2_preproc

# ---- quiet TF logs (Cloud friendlier) ----
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ---------- constants ----------
APP_DIR = Path(__file__).parent.resolve()
KERAS_PATH = APP_DIR / "brain_effv2b0_infer.keras"
LABELS_PATH = APP_DIR / "labels.json"
SAVEDMODEL_DIR = APP_DIR / "brain_savedmodel"   # contains saved_model.pb if used
IMG_SZ = 300

st.set_page_config(page_title="Brain MRI Classifier", layout="wide")
st.title("🧠 Brain MRI — Alzheimer’s Stage Classifier")

# ---------- utilities ----------
def load_labels(path: Path):
    lab = json.loads(path.read_text(encoding="utf-8"))
    # normalize to list index->name
    return [lab[str(i)] if str(i) in lab else lab[i] for i in range(len(lab))]

def preprocess_pil(img: Image.Image) -> np.ndarray:
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
    # Only available when loading the .keras model (SavedModel wrapper has no layers)
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
    weights = tf.reduce_mean(grads, axis=(0,1))
    cam = tf.reduce_sum(tf.multiply(weights, conv_out[0]), axis=-1)
    cam = tf.maximum(cam, 0) / (tf.reduce_max(cam) + 1e-8)
    cam = tf.image.resize(cam[..., None], (IMG_SZ, IMG_SZ)).numpy().squeeze()

    base = np.array(img.convert("RGB").resize((IMG_SZ, IMG_SZ)), np.float32) / 255.0
    heat = plt.cm.jet(cam)[..., :3]
    overlay = np.clip((1 - alpha) * base + alpha * heat, 0, 1)
    return (overlay * 255).astype(np.uint8)

# ---------- load model & labels ----------
try:
    class_names = load_labels(LABELS_PATH)
except Exception as e:
    st.error(f"Could not read labels.json: {e}")
    st.stop()

# Try .keras first (needs exact custom object key used during save),
# else fall back to SavedModel.
try:
    if KERAS_PATH.exists():
        # IMPORTANT: your .keras contains a registered Lambda named "custom>effv2_preproc"
        model = load_model(str(KERAS_PATH),
                           custom_objects={"custom>effv2_preproc": v2_preproc})
        is_keras = True
        st.success("Loaded Keras model.")
    elif (SAVEDMODEL_DIR / "saved_model.pb").exists():
        infer = tf.saved_model.load(str(SAVEDMODEL_DIR))
        serving = infer.signatures["serving_default"]
        class Wrap:
            def predict(self, x, verbose=0):
                out = serving(tf.constant(x))
                return next(iter(out.values())).numpy()
            @property
            def layers(self):  # for Grad-CAM guard
                return []
        model = Wrap()
        is_keras = False
        st.success("Loaded SavedModel.")
    else:
        raise FileNotFoundError("No model found. Place 'brain_effv2b0_infer.keras' "
                                "or 'brain_savedmodel/' next to this file.")
except Exception as e:
    st.error(f"Model load error: {e}")
    st.stop()

st.caption("Classes: " + " | ".join(class_names))
st.markdown("---")

# ---------- UI: upload & predict (robust on Streamlit Cloud) ----------
show_cam = st.checkbox("Show Grad-CAM (only for .keras models)", value=True)
uploaded = st.file_uploader("Upload an MRI image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    # Robust read for Cloud: use bytes → PIL → NumPy for st.image
    try:
        data = uploaded.getvalue()
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        st.error(f"Could not open image: {e}")
        st.stop()

    st.image(np.array(img), caption="Input", use_container_width=True)

    try:
        probs = predict_one(model, img)[0]
    except Exception as e:
        st.error(f"Inference failed: {e}")
        st.stop()

    top = int(np.argmax(probs))
    st.subheader(f"Predicted Stage: **{class_names[top]}** (confidence {probs[top]:.3f})")

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(range(len(class_names)), probs)
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    st.pyplot(fig, use_container_width=True)

    if show_cam and is_keras:
        overlay = grad_cam(model, img)
        if overlay is not None:
            st.image(overlay, caption="Grad-CAM overlay", use_container_width=True)
    elif show_cam and not is_keras:
        st.caption("Grad-CAM is only available when loading the .keras model.")
