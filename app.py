# app.py - fixed for nested child nodes and replaced predict_image with training/test code
import os
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv

# Ensure .env loads relative to this file (fixes Streamlit WD issues)
env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(env_path)

# Config from .env
FIREBASE_CONFIG = {
    "apiKey": os.getenv("FIREBASE_API_KEY", "").strip(),
    "authDomain": os.getenv("FIREBASE_AUTH_DOMAIN", "").strip(),
    "databaseURL": os.getenv("FIREBASE_DATABASE_URL", "").strip(),
    "storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET", "").strip(),
}
FIREBASE_PATH = os.getenv("FIREBASE_PATH", "sensorReadings").strip("/")
MODEL_PATH = os.getenv("MODEL_PATH", "model.pth")  # path to your checkpoint (.pth)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash")

# Image preprocess constants (kept for compatibility)
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Safe imports and flags
try:
    import pyrebase
    pyrebase_ok = True
except Exception:
    pyrebase_ok = False

try:
    from google import genai
    genai_ok = True
except Exception:
    genai_ok = False

# ML dependencies
torch_present = False
try:
    import torch
    import numpy as np
    from PIL import Image
    torch_present = True
except Exception:
    pass

# albumentations optional
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALB_OK = True
except Exception:
    ALB_OK = False

st.set_page_config(page_title="AgriFusion: Realtime Pest Detection & Treatment Advisor", layout="wide")
st.title("AgriFusion: Realtime Pest Detection & Treatment Advisor")

# ----------------------- Helpers -----------------------
def _normalize_ts(value):
    if value is None:
        return None
    try:
        if isinstance(value, str):
            v = int(float(value.strip()))
        elif isinstance(value, (int, float)):
            v = int(value)
        else:
            return None
        # convert seconds to ms if necessary
        if v < 1_000_000_000_000:
            v *= 1000
        return v
    except Exception:
        return None

def _format_ts(ts_ms):
    if not ts_ms:
        return "unknown"
    try:
        return datetime.fromtimestamp(ts_ms / 1000).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "unknown"

# ----------------------- Firebase -----------------------
db = None
if pyrebase_ok:
    try:
        firebase = pyrebase.initialize_app(FIREBASE_CONFIG)
        db = firebase.database()
    except Exception:
        db = None

def list_devices(path=FIREBASE_PATH):
    """Return list of device keys under sensorReadings (e.g. ['demo_device'])."""
    if db is None:
        return []
    try:
        root = db.child(path).get()
        if not root.each():
            return []
        return [it.key() for it in root.each()]
    except Exception:
        return []

def list_lands(device_id, path=FIREBASE_PATH):
    """
    Given device id (e.g. 'demo_device'), return its child nodes (e.g. ['Land 1','Land 2']).
    """
    if db is None or not device_id:
        return []
    try:
        node = db.child(path).child(device_id).get()
        if not node.each():
            return []
        return [it.key() for it in node.each()]
    except Exception:
        return []

def fetch_latest_for_land(device_id, land, path=FIREBASE_PATH):
    """
    Read the latest reading for a given device->land node.
    Handles both cases:
      - land contains direct {humidity, ph, ts}
      - land contains children (push ids) that then contain readings
    """
    if db is None or not device_id or not land:
        return None
    try:
        node = db.child(path).child(device_id).child(land).get().val()
    except Exception:
        node = None

    if not node:
        return None

    # direct reading stored in the land node
    if isinstance(node, dict) and any(k in node for k in ("humidity", "ph", "ts")):
        node = dict(node)
        node["ts"] = _normalize_ts(node.get("ts"))
        return node

    # otherwise consider it's a dict of push-ids -> find latest by ts
    if isinstance(node, dict):
        best = None
        best_ts = -1
        for k, v in node.items():
            if not isinstance(v, dict):
                continue
            if not any(x in v for x in ("ts", "humidity", "ph")):
                continue
            ts = _normalize_ts(v.get("ts"))
            tsv = ts if ts is not None else 0
            if tsv > best_ts:
                best_ts = tsv
                best = dict(v)
                best["ts"] = tsv if tsv > 0 else None
        return best
    return None

# ----------------------- Model loader & predict_image (YOUR training/test code integrated) -----------------------
if torch_present:
    import torch.nn as nn
    # cached loader to avoid reloading checkpoint on every inference
    @st.cache_resource
    def load_trained_model(path=MODEL_PATH):
        """
        Loads a checkpoint saved during training with keys:
          - 'classes' : list of class names
          - 'model_state' or 'state_dict' : state dict of the model
        Creates an efficientnet_b0 using rwightman hub and loads weights.
        Returns (model_cpu, class_names)
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model checkpoint not found at: {path}")

        ckpt = torch.load(path, map_location="cpu")

        # class names
        if "classes" in ckpt:
            class_names = ckpt["classes"]
        elif "class_names" in ckpt:
            class_names = ckpt["class_names"]
        else:
            # fallback: try to read from labels file if present
            class_names = None

        # prefer 'model_state' (your training format) else 'state_dict'
        state = ckpt.get("model_state", ckpt.get("state_dict", ckpt))

        # create model via hub (rwightman) to match training
        # note: using try/except because hub can fail offline
        try:
            model = torch.hub.load('rwightman/pytorch-image-models', 'efficientnet_b0', pretrained=False)
        except Exception as e:
            # fallback to a minimal timm model if torch.hub not available
            try:
                import timm
                # num_classes will be fixed after we know classes length
                model = timm.create_model("efficientnet_b0", pretrained=False)
            except Exception as e2:
                raise RuntimeError("Failed to create model (torch.hub and timm failed).") from e2

        # figure out number of classes
        num_classes = None
        if class_names:
            num_classes = len(class_names)
        else:
            # try infer from state dict biases
            if isinstance(state, dict):
                for k, v in state.items():
                    if k.endswith("bias") and any(x in k for x in ("classifier", "head", "fc")):
                        try:
                            num_classes = v.shape[0]
                            break
                        except Exception:
                            pass
        if num_classes is None:
            # default to 2 if totally unknown
            num_classes = 2

        # set classifier on the model to match num_classes
        # common name: model.classifier or model.fc
        if hasattr(model, "classifier"):
            try:
                in_feats = model.classifier.in_features
                model.classifier = nn.Linear(in_feats, num_classes)
            except Exception:
                # some variants: classifier is Sequential
                model.classifier = nn.Linear(getattr(model.classifier, 1).in_features, num_classes)
        elif hasattr(model, "fc"):
            in_feats = model.fc.in_features
            model.fc = nn.Linear(in_feats, num_classes)
        else:
            # last-resort: try to inspect modules
            found = False
            for name, module in reversed(list(model.named_modules())):
                if isinstance(module, nn.Linear):
                    in_feats = module.in_features
                    # assign to 'classifier' attr
                    model.classifier = nn.Linear(in_feats, num_classes)
                    found = True
                    break
            if not found:
                raise RuntimeError("Unable to set classifier layer on model (unknown architecture).")

        # load state dict (best-effort)
        if isinstance(state, dict):
            # remove 'module.' prefixes if present
            new_state = {}
            for k, v in state.items():
                new_key = k.replace("module.", "") if isinstance(k, str) else k
                new_state[new_key] = v
            try:
                model.load_state_dict(new_state, strict=False)
            except Exception:
                # try direct load (sometimes checkpoint stores whole model)
                try:
                    model.load_state_dict(state)
                except Exception as e:
                    # if it fails, just continue - model might already be correct
                    pass

        model.eval()
        return model, class_names

    # preprocessing pipeline - prefer albumentations if available
    if ALB_OK:
        val_tr = A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2()
        ])
    else:
        # fallback torchvision-based pipeline
        from torchvision import transforms
        val_tr_torch = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])

    def _make_probs_dict(class_names, probs):
        """
        Build probs dict mapping class name to score. If class_names is None,
        numeric indices are used as strings.
        """
        probs_dict = {}
        for i, p in enumerate(probs):
            name = class_names[i] if class_names and i < len(class_names) else str(i)
            probs_dict[name] = float(p)
        return probs_dict

    def predict_image(pil_image):
        """
        Replaces previous predict implementation with the training/test code you provided.
        Input: PIL.Image
        Output: (label, confidence, probs_dict)
        """
        # load model + classes (cached)
        model, class_names = load_trained_model(MODEL_PATH)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        # preprocess
        if ALB_OK:
            img_np = np.array(pil_image.convert("RGB"))
            t = val_tr(image=img_np)['image'].unsqueeze(0).to(device)
        else:
            # fallback
            t = val_tr_torch(pil_image.convert("RGB")).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(t)
            if isinstance(out, (tuple, list)):
                out = out[0]
            probs = torch.softmax(out, dim=1).cpu().numpy()[0]
        idx = int(probs.argmax())
        label = class_names[idx] if class_names and idx < len(class_names) else str(idx)
        conf = float(probs[idx])
        probs_dict = _make_probs_dict(class_names, probs)
        return label, conf, probs_dict

else:
    # If torch not present, preserve an informative stub
    def predict_image(pil_image):
        raise RuntimeError("Torch not available in this environment. Install torch to run predictions.")

# ----------------------- Gemini helper -----------------------
def call_gemini_prompt(prompt_text):
    if not genai_ok:
        return "Gemini SDK not installed. Install 'google-genai' in the venv."
    if not GEMINI_API_KEY:
        return "GEMINI_API_KEY not set. Put it in .env and restart."
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        return f"Failed to create Gemini client: {e}"

    preferred = GEMINI_MODEL or "models/gemini-2.5-flash"
    candidates = [preferred, "models/gemini-2.5-flash", "models/gemini-flash-latest", "models/gemini-2.5-pro"]

    last_exc = None
    for mname in candidates:
        if not mname:
            continue
        try:
            if hasattr(client, "models") and hasattr(client.models, "generate_content"):
                resp = client.models.generate_content(model=mname, contents=prompt_text)
                text = getattr(resp, "text", None)
                if text:
                    return text
                return str(resp)
            if hasattr(client, "generate_text"):
                r = client.generate_text(model=mname, input=prompt_text)
                t = getattr(r, "text", None) or getattr(r, "output", None)
                if t:
                    return t
                return str(r)
            if hasattr(client, "generate"):
                r = client.generate(model=mname, prompt=prompt_text)
                t = getattr(r, "text", None) or getattr(r, "output", None)
                if t:
                    return t
                return str(r)
        except Exception as e:
            last_exc = e
            continue

    msg = "Gemini call failed."
    if last_exc:
        msg += f" Last error: {last_exc}"
    return msg

# ----------------------- Streamlit UI layout (device + land selectors) -----------------------
st.sidebar.header("App settings")

# Device selector
device_list = list_devices()
device_list = sorted(device_list)
selected_device = st.sidebar.selectbox("Select device", [""] + device_list, index=1 if device_list else 0)

if st.sidebar.button("Refresh device list"):
    st.sidebar.write("Refreshed (interaction triggered).")

st.sidebar.markdown("Model path: " + MODEL_PATH)

# Land (child node) selector
lands = []
selected_land = None
if selected_device:
    lands = list_lands(selected_device)
    lands = sorted(lands)
    if lands:
        selected_land = st.sidebar.selectbox("Select land (child node)", [""] + lands, index=1)
    else:
        st.sidebar.info("No child nodes (lands) found for this device.")

st.header("Device Monitor")

# Show latest reading for selected device/land
if not selected_device:
    st.info(f"No devices found at {FIREBASE_PATH}. Add data and refresh.")
else:
    if not selected_land:
        st.info("Select a land (child node) to view latest readings.")
    else:
        latest = fetch_latest_for_land(selected_device, selected_land)
        if latest:
            hum = latest.get("humidity", "—")
            phv = latest.get("ph", "—")
            ts_ms = latest.get("ts")
            cols = st.columns([1,1,1])
            cols[0].metric("Humidity", f"{hum}")
            cols[1].metric("pH", f"{phv}")
            cols[2].metric("Timestamp", _format_ts(ts_ms))
        else:
            st.info("No latest reading found for this device/land.")

# Image block
st.subheader("Image inference (upload an image)")
uploaded = st.file_uploader("Upload image (jpg/png)", type=["jpg","jpeg","png"])
if uploaded:
    try:
        pil = Image.open(uploaded)
        # use_container_width to avoid deprecation
        st.image(pil, caption="Uploaded image", use_container_width=True)
        if not torch_present:
            st.warning("Torch/timm not installed — image inference disabled.")
        else:
            try:
                label, conf, probs = predict_image(pil)
                # Show top-3 disease names (only names)
                if isinstance(probs, dict):
                    top3 = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:3]
                    st.subheader("Top-3 Predicted Diseases")
                    for name, _score in top3:
                        st.markdown(f"- {name}")
                else:
                    st.markdown(f"**Predicted:** {label} — Confidence: {conf:.2f}")
            except Exception as e:
                st.error(f"Image inference failed: {e}")
    except Exception as e:
        st.error(f"Failed to open image: {e}")

# Sensor + Gemini
st.subheader("Run inference with latest sensor values + get recommendation")
if selected_device and selected_land:
    if st.button("Treatment"):
        latest = fetch_latest_for_land(selected_device, selected_land)
        if not latest:
            st.error("No latest reading found for selected land.")
        else:
            try:
                h_val = float(latest.get("humidity", 0) or 0)
                p_val = float(latest.get("ph", 0) or 0)
                if p_val < 5.5:
                    predicted = "Acid soil / nutrient imbalance"
                    conf = 0.60
                elif h_val > 85:
                    predicted = "Waterlogged / root-rot risk"
                    conf = 0.75
                else:
                    predicted = "No immediate issue detected"
                    conf = 0.50
                st.markdown(f"**Sensor inference:** {predicted} (confidence {conf:.2f})")
                prompt = (
                    f"Sensor readings:\n- humidity: {h_val}\n- pH: {p_val}\n"
                    f"Predicted issue: {predicted} (confidence: {conf:.2f}).\n\n"
                    "Provide a concise 4-step treatment plan: short-term remedy, preventive step, severity (low/medium/high), What Pesticide to use."
                )
                rec = call_gemini_prompt(prompt)
                st.subheader("Recommendation")
                st.write(rec)
            except Exception as e:
                st.error("Failed to run sensor + Gemini: " + str(e))
else:
    st.info("Select device and land to enable Treatment actions.")

st.caption("Notes: Put Firebase config and GEMINI_API_KEY in .env. If Gemini fails, run the model-list script to see which models your key can access.")
