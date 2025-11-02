# app.py
# Brain MRI — Alzheimer’s Stage Classifier (EffNetV2-B0, 300x300)
# Auto-discovers a model (.keras / .h5 / SavedModel) + labels.json anywhere in repo.

import os, io, json, base64
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as v2_preproc

# Make TF quieter on Cloud
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

APP_DIR = Path(__file__).parent.resolve()
IMG_SZ = 300

st.set_page_config(page_title="Brain MRI Classifier", layout="wide")
st.title("🧠 Brain MRI — Alzheimer’s Stage Classifier")

# ---------- small helpers ----------
def is_git_lfs_pointer(p: Path) -> bool:
    try:
        if p.is_file() and p.stat().st_size <= 2048:
            head = p.read_text(errors="ignore")
            return head.startswith("version https://git-lfs.github.com/spec")
    except Exception:
        pass
    return False

def load_labels(path: Path) -> List[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        max_idx = max(int(k) for k in data.keys())
        return [data[str(i)] for i in range(max_idx + 1)]
    if isinstance(data, list):
        return data
    raise ValueError("labels.json must be a dict or a list")

def pil_to_png_bytes(pil_img: Image.Image) -> bytes:
    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()

def show_image_html(pil_img: Image.Image, caption: str = ""):
    b64 = base64.b64encode(pil_to_png_bytes(pil_img)).decode("ascii")
    cap = f"<div style='margin-top:6px;color:#999;font-size:0.9em'>{caption}</div>" if caption else ""
    st.markdown(
        f"<div><img src='data:image/png;base64,{b64}' style='max-width:100%;height:auto;border-radius:8px;'/>"
        f"{cap}</div>",
        unsafe_allow_html=True,
    )

def preprocess_pil(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((IMG_SZ, IMG_SZ))
    x = np.array(img, dtype=np.float32)[None, ...]
    return v2_preproc(x)

def predict_one(model, img: Image.Image) -> np.ndarray:
    x = preprocess_pil(img)
    return model.predict(x, verbose=0)

def get_last_conv_name(keras_model) -> Optional[str]:
    last = None
    for lyr in getattr(keras_model, "layers", []):
        if isinstance(lyr, tf.keras.layers.Conv2D):
            last = lyr.name
    return last

def grad_cam(keras_model, img: Image.Image, alpha=0.40):
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
        loss = preds[:, tf.argmax(preds[0])]
    grads = tape.gradient(loss, conv_out)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = tf.reduce_sum(tf.multiply(weights, conv_out[0]), axis=-1)
    cam = tf.maximum(cam, 0) / (tf.reduce_max(cam) + 1e-8)
    cam = tf.image.resize(cam[..., None], (IMG_SZ, IMG_SZ)).numpy().squeeze()
    base = np.asarray(img.convert("RGB").resize((IMG_SZ, IMG_SZ)), dtype=np.float32) / 255.0
    heat = plt.cm.jet(cam)[..., :3]
    overlay = np.clip((1 - alpha) * base + alpha * heat, 0, 1)
    return (overlay * 255).astype(np.uint8)

# ---------- discovery ----------
def find_labels_json(start: Path) -> Path:
    # Prefer root
    if (start / "labels.json").exists():
        return start / "labels.json"
    for p in start.rglob("labels.json"):
        return p
    raise FileNotFoundError("labels.json not found in repository.")

def model_candidates(start: Path) -> List[Tuple[str, Path]]:
    cands: List[Tuple[str, Path]] = []
    # Prefer specific filename in root
    p = start / "brain_effv2b0_infer.keras"
    if p.exists() and not is_git_lfs_pointer(p):
        cands.append(("keras", p))

    # Any .keras (largest first, skip LFS pointers)
    for q in sorted(start.rglob("*.keras"),
                    key=lambda x: x.stat().st_size if x.exists() else 0,
                    reverse=True):
        if q.resolve() != p.resolve() and not is_git_lfs_pointer(q):
            cands.append(("keras", q))

    # Any .h5
    for q in sorted(start.rglob("*.h5"),
                    key=lambda x: x.stat().st_size if x.exists() else 0,
                    reverse=True):
        if not is_git_lfs_pointer(q):
            cands.append(("h5", q))

    # Any SavedModel folder
    for pb in start.rglob("saved_model.pb"):
        cands.append(("savedmodel", pb.parent))

    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for kind, path in cands:
        key = (kind, str(path.resolve()))
        if key not in seen:
            seen.add(key)
            uniq.append((kind, path))
    return uniq

@st.cache_resource(show_spinner=True)
def cached_load_assets(start: Path):
    labels_path = find_labels_json(start)
    labels = load_labels(labels_path)

    errs = []
    for kind, path in model_candidates(start):
        try:
            if kind in ("keras", "h5"):
                custom = {"custom>effv2_preproc": v2_preproc, "preprocess_input": v2_preproc}
                try:
                    model = load_model(str(path), custom_objects=custom)
                except TypeError:
                    model = load_model(str(path))
                return model, True, labels, {"model_kind": kind, "model_path": str(path.relative_to(start)),
                                             "labels_path": str(labels_path.relative_to(start))}
            else:
                infer = tf.saved_model.load(str(path))
                serving = infer.signatures["serving_default"]

                class Wrap:
                    def predict(self, x, verbose=0):
                        out = serving(tf.constant(x))
                        return next(iter(out.values())).numpy()
                    @property
                    def layers(self):
                        return []

                return Wrap(), False, labels, {"model_kind": kind, "model_path": str(path.relative_to(start)),
                                               "labels_path": str(labels_path.relative_to(start))}
        except Exception as e:
            errs.append(f"[{kind}] {path}: {e}")

    # If nothing worked:
    more = "\n".join(errs) if errs else "No model files found."
    raise RuntimeError(
        "Could not load any model. Tried (in order): brain_effv2b0_infer.keras, *.keras, *.h5, SavedModel.\n"
        + more + "\nIf your .keras is a Git-LFS pointer, enable LFS on Streamlit Cloud or upload the real file."
    )

# ---------- load everything ----------
try:
    model, is_keras_graph, class_names, meta = cached_load_assets(APP_DIR)
    with st.expander("Loaded assets (auto-discovered)", expanded=False):
        st.code(json.dumps(meta, indent=2))
    st.success(f"Model loaded: {meta['model_kind']} → {meta['model_path']}")
except Exception as e:
    st.error(f"{e}")
    st.stop()

# Sanity: classes vs labels length
try:
    dummy = v2_preproc(np.zeros((1, IMG_SZ, IMG_SZ, 3), dtype=np.float32))
    out = model.predict(dummy, verbose=0)
    if out.shape[-1] != len(class_names):
        st.warning(f"Model outputs {out.shape[-1]} classes; labels.json has {len(class_names)}.")
except Exception:
    pass

st.caption("Classes: " + " | ".join(class_names))
st.markdown("---")

# ---------- UI ----------
show_cam = st.checkbox("Show Grad-CAM (only for .keras/.h5 models)", value=True)
uploaded = st.file_uploader("Upload an MRI image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    try:
        img = Image.open(io.BytesIO(uploaded.getvalue())).convert("RGB")
    except Exception as e:
        st.error(f"Could not open image: {e}")
        st.stop()

    show_image_html(img, caption="Input")

    try:
        probs = predict_one(model, img)[0]
    except Exception as e:
        st.error(f"Inference failed: {e}")
        st.stop()

    idx = int(np.argmax(probs))
    st.subheader(f"Predicted Stage: **{class_names[idx]}** (confidence {probs[idx]:.3f})")

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(range(len(class_names)), probs)
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    if show_cam and is_keras_graph:
        cam = grad_cam(model, img)
        if cam is not None:
            show_image_html(Image.fromarray(cam), caption="Grad-CAM overlay")
        else:
            st.caption("Grad-CAM not available for this model.")
    elif show_cam and not is_keras_graph:
        st.caption("Grad-CAM is only available for .keras/.h5 Keras models.")
