# app.py
# Brain MRI — Alzheimer’s Stage Classifier (EffNetV2-B0, 300x300)
# Put these next to this file:
#   - brain_effv2b0_infer.keras   (preferred)  OR  brain_savedmodel/ (fallback)
#   - labels.json
#
# Optional: set st.secrets["MODEL_URL"] to a direct URL of the .keras file.
# If present, the app will download the model at first run.

from pathlib import Path
import json
import os
import urllib.request

import numpy as np
from PIL import Image

import streamlit as st
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as v2_preproc

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
    # support {"0":"CN","1":"MCI",...} or ["CN","MCI",...]
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
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = tf.reduce_sum(tf.multiply(weights, conv_out[0]), axis=-1)
    cam = tf.maximum(cam, 0) / (tf.reduce_max(cam) + 1e-8)
    cam = tf.image.resize(cam[..., None], (IMG_SZ, IMG_SZ)).numpy().squeeze()

    base = np.array(img.convert("RGB").resize((IMG_SZ, IMG_SZ)), np.float32) / 255.0
    heat = plt.cm.jet(cam)[..., :3]
    overlay = np.clip((1 - alpha) * base + alpha * heat, 0, 1)
    return (overlay * 255).astype(np.uint8)

def to_display_rgb(pil: Image.Image) -> Image.Image:
    """Convert any PIL image (incl. 16-bit, float, palette, CMYK) to 8-bit RGB for safe Streamlit display."""
    if pil.mode in ("I;16", "I", "F"):  # 16-bit/float grayscale
        arr = np.array(pil, dtype=np.float32)
        if np.isfinite(arr).any():
            arr = arr - float(np.nanmin(arr))
            mx = float(np.nanmax(arr))
            if mx > 0:
                arr = arr / mx
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="L").convert("RGB")
    if pil.mode == "RGBA":
        bg = Image.new("RGB", pil.size, (255, 255, 255))
        bg.paste(pil, mask=pil.split()[-1])
        return bg
    if pil.mode in ("RGB", "L"):
        return pil.convert("RGB")
    return pil.convert("RGB")

def ensure_model_present():
    """Ensure either .keras or SavedModel exists; optionally download .keras via secrets URL."""
    if KERAS_PATH.exists() or (SAVEDMODEL_DIR / "saved_model.pb").exists():
        return
    model_url = st.secrets.get("MODEL_URL")
    if model_url:
        KERAS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with st.spinner("Downloading model…"):
            urllib.request.urlretrieve(model_url, KERAS_PATH)
    # show contents for debugging if still missing
    if not KERAS_PATH.exists() and not (SAVEDMODEL_DIR / "saved_model.pb").exists():
        st.write("App folder contents:", sorted(p.name for p in APP_DIR.iterdir()))
        raise FileNotFoundError(
            "No model found. Add 'brain_effv2b0_infer.keras' or 'brain_savedmodel/' next to app.py "
            "or set st.secrets['MODEL_URL'] to a direct .keras download link."
        )

# ---------- load labels ----------
try:
    class_names = load_labels(LABELS_PATH)
except Exception as e:
    st.error(f"Could not read labels.json: {e}")
    st.stop()

# ---------- ensure model + load ----------
try:
    ensure_model_present()
    if KERAS_PATH.exists():
        # NB: custom_objects key must match what you used when saving; safe to alias preprocess
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
            def layers(self):  # prevent Grad-CAM usage on SavedModel
                return []

        model = Wrap()
        is_keras = False
        st.success("Loaded SavedModel.")
    else:
        # unreachable because ensure_model_present would have raised, but keep for safety
        raise FileNotFoundError("Model not found.")
except Exception as e:
    st.error(f"Model load error: {e}")
    st.stop()

st.caption("Classes: " + " | ".join(class_names))
st.markdown("---")

# ---------- UI: upload & predict ----------
show_cam = st.checkbox("Show Grad-CAM (only for .keras models)", value=True)
file = st.file_uploader("Upload an MRI image (JPG/PNG/TIFF)", type=["jpg", "jpeg", "png", "tif", "tiff"])

if file:
    raw_img = Image.open(file)
    disp_img = to_display_rgb(raw_img)

    # Safe display (Cloud sometimes raises TypeError on odd modes)
    try:
        st.image(disp_img, caption="Input", use_container_width=True)
    except TypeError:
        st.image(np.array(disp_img), caption="Input", use_container_width=True)

    try:
        probs = predict_one(model, raw_img)[0]
    except Exception as e:
        st.error(f"Inference error: {e}")
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
        overlay = grad_cam(model, raw_img)
        if overlay is not None:
            st.image(overlay, caption="Grad-CAM overlay", use_container_width=True)
