import streamlit as st
import html
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from pathlib import Path

# -------------------- Page config --------------------
st.set_page_config(page_title="Driver Fatigue Monitor", page_icon="🚗", layout="wide")
ROOT = Path(__file__).parent
IMG_SIZE = (224, 224)
CLASS_NAMES = ["Closed", "Open", "no_yawn", "yawn"]
FATIGUE_MAP = {"Open": 0, "no_yawn": 0, "yawn": 1, "Closed": 2}
FATIGUE_NAMES = ["Alert", "Mild Fatigue", "Severe Fatigue"]
MODEL_FILES = {
    "MobileNetV2 (Recommended)": ROOT / "driver_drowsiness_mobilenetv2.keras",
    "Custom CNN": ROOT / "driver_drowsiness_custom_cnn.keras",
}


def render_html_table(frame):
    """Render a small table without Streamlit's Arrow/Altair dataframe stack."""
    safe = frame.copy()
    headers = "".join(f"<th>{html.escape(str(c))}</th>" for c in safe.columns)
    rows = []
    for _, row in safe.iterrows():
        cells = "".join(f"<td>{html.escape(str(v))}</td>" for v in row)
        rows.append(f"<tr>{cells}</tr>")
    body = "".join(rows)
    st.markdown(
        "<div style='overflow-x:auto'><table style='width:100%;border-collapse:collapse'>"
        f"<thead><tr>{headers}</tr></thead><tbody>{body}</tbody></table></div>",
        unsafe_allow_html=True,
    )

# -------------------- Styling --------------------
st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
[data-testid="stMetric"] {background: rgba(120, 130, 150, .09); padding: 14px 16px; border-radius: 12px;}
div[data-testid="stSidebar"] {border-right: 1px solid rgba(120,130,150,.2);}
</style>
""", unsafe_allow_html=True)

# -------------------- Loaders --------------------
@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(model_path, compile=False)

@st.cache_data
def load_comparison():
    path = ROOT / "model_comparison.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame({
        "Model": ["Custom CNN", "MobileNetV2"],
        "Test accuracy": [0.250575, 0.852874],
        "Training time (seconds)": [49.097346, 23.799700],
    })

@st.cache_data
def load_progression():
    path = ROOT / "fatigue_progression_results.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame({
        "Minute": list(range(1,16)),
        "Mean fatigue score": [0.666667,0.8,0.666667,0.233333,0.7,0.8,0.7,0.566667,0.8,0.266667,0.533333,0.733333,0.866667,0.333333,0.466667],
        "Severe frames": [6,10,8,3,8,11,9,6,9,4,7,9,12,4,3],
        "Mild frames": [8,4,4,1,5,2,3,5,6,0,2,4,2,2,1],
        "Alert frames": [16,16,18,26,17,17,18,19,15,26,21,17,16,24,11],
        "Stage": ["Mild Fatigue","Mild Fatigue","Mild Fatigue","Alert","Mild Fatigue","Mild Fatigue","Mild Fatigue","Mild Fatigue","Mild Fatigue","Alert","Mild Fatigue","Mild Fatigue","Mild Fatigue","Alert","Alert"],
    })

def preprocess_image(pil_image):
    image = pil_image.convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(image).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)

def predict_image(model, pil_image):
    batch = preprocess_image(pil_image)
    probs = model.predict(batch, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    predicted_class = CLASS_NAMES[pred_idx]
    fatigue_stage = FATIGUE_NAMES[FATIGUE_MAP[predicted_class]]
    return predicted_class, float(probs[pred_idx]), probs, fatigue_stage

def stage_color(stage):
    if stage == "Alert":
        return "🟢"
    if stage == "Mild Fatigue":
        return "🟠"
    return "🔴"

# -------------------- Sidebar navigation --------------------
st.sidebar.title("🚗 Driver Fatigue Monitor")
page = st.sidebar.radio("Navigation", ["Dashboard", "Predict", "Model Performance", "Fatigue Analysis", "About"])
st.sidebar.caption("Image-based drowsiness and fatigue classification")
st.sidebar.markdown("---")
st.sidebar.info("This is an educational prototype. It is not a certified safety system and should not be used to make driving-safety decisions.")

# -------------------- Header --------------------
st.title("🚗 Driver Fatigue Monitor")
st.caption("Computer vision prototype for eye-state and yawning classification")
st.markdown("---")

comparison = load_comparison()
progression = load_progression()

if page == "Dashboard":
    st.subheader("Project Overview")
    st.write("This prototype classifies driver images into four visual states and maps those states into three fatigue levels.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset images", "2,900")
    c2.metric("Classes", "4")
    c3.metric("Best test accuracy", "85.29%")
    c4.metric("Best model", "MobileNetV2")

    st.markdown("### Classification labels")
    cards = st.columns(4)
    descriptions = {
        "Closed": "Eyes closed → Severe Fatigue",
        "Open": "Eyes open → Alert",
        "no_yawn": "No yawn detected → Alert",
        "yawn": "Yawning → Mild Fatigue",
    }
    for col, cls in zip(cards, CLASS_NAMES):
        col.markdown(f"**{cls}**")
        col.caption(descriptions[cls])

    st.markdown("### Dataset distribution")
    counts = pd.Series({"Closed":726, "Open":726, "no_yawn":725, "yawn":723})
    fig, ax = plt.subplots(figsize=(8,4))
    counts.plot(kind="bar", ax=ax)
    ax.set_title("Image count per class")
    ax.set_xlabel("Class")
    ax.set_ylabel("Images")
    ax.tick_params(axis="x", rotation=0)
    st.pyplot(fig, clear_figure=True)

    st.markdown("### Model comparison")
    display_df = comparison.copy()
    if "Test accuracy" in display_df:
        display_df["Test accuracy"] = (display_df["Test accuracy"] * 100).round(2).astype(str) + "%"
    render_html_table(display_df)

elif page == "Predict":
    st.subheader("Image Prediction")
    st.write("Upload an image or capture one using your device camera. The selected model predicts an eye/yawn class.")
    model_choice = st.selectbox("Choose model", list(MODEL_FILES.keys()), index=0)
    input_mode = st.radio("Image source", ["Upload image", "Camera"], horizontal=True)
    image_file = None
    if input_mode == "Upload image":
        image_file = st.file_uploader("Upload JPG, JPEG or PNG", type=["jpg","jpeg","png"])
    else:
        image_file = st.camera_input("Capture a driver image")

    if image_file is not None:
        pil_img = Image.open(image_file).convert("RGB")
        st.image(pil_img, caption="Input image", width=400)
        if st.button("Run prediction", type="primary"):
            model_path = MODEL_FILES[model_choice]
            if not model_path.exists():
                st.error(f"Model file missing: {model_path.name}")
            else:
                with st.spinner("Loading model and analyzing image..."):
                    model = load_model(str(model_path))
                    pred_class, confidence, probs, fatigue_stage = predict_image(model, pil_img)
                st.markdown("## Prediction result")
                r1, r2, r3 = st.columns(3)
                r1.metric("Predicted class", pred_class)
                r2.metric("Confidence", f"{confidence*100:.2f}%")
                r3.metric("Fatigue stage", fatigue_stage)
                st.progress(float(confidence))
                st.markdown("### Class probabilities")
                prob_df = pd.DataFrame({"Class": CLASS_NAMES, "Probability": probs, "Percentage": [f"{p*100:.2f}%" for p in probs]})
                fig, ax = plt.subplots(figsize=(7,3.5))
                ax.bar(prob_df["Class"], prob_df["Probability"])
                ax.set_ylim(0, 1)
                ax.set_ylabel("Probability")
                ax.set_title("Prediction probability by class")
                ax.tick_params(axis="x", rotation=15)
                st.pyplot(fig, clear_figure=True)
                render_html_table(prob_df[["Class","Percentage"]])
                st.warning("Single-image predictions are not a validated measure of a driver's real fatigue or alertness. Use this only for demonstration.")

elif page == "Model Performance":
    st.subheader("Model Performance")
    if not comparison.empty:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Test accuracy")
            fig, ax = plt.subplots(figsize=(6,4))
            acc = comparison.copy()
            acc["Accuracy (%)"] = acc["Test accuracy"] * 100
            ax.bar(acc["Model"], acc["Accuracy (%)"])
            ax.set_ylabel("Accuracy (%)")
            ax.set_ylim(0,100)
            ax.set_title("Test accuracy comparison")
            st.pyplot(fig, clear_figure=True)
        with c2:
            st.markdown("#### Training time")
            fig, ax = plt.subplots(figsize=(6,4))
            ax.bar(comparison["Model"], comparison["Training time (seconds)"])
            ax.set_ylabel("Seconds")
            ax.set_title("Training time comparison")
            st.pyplot(fig, clear_figure=True)

        st.markdown("### Results table")
        view = comparison.copy()
        view["Test accuracy"] = (view["Test accuracy"]*100).round(2).astype(str) + "%"
        view["Training time (seconds)"] = view["Training time (seconds)"].round(2)
        render_html_table(view)

        st.markdown("### Interpretation")
        st.success("MobileNetV2 performed better on the held-out test set in this experiment.")
        st.write("MobileNetV2 achieved about 85.29% test accuracy compared with about 25.06% for the custom CNN. It also trained faster in the recorded run (about 23.80 seconds vs 49.10 seconds).")
        st.warning("The MobileNetV2 confusion matrix shows a notable weakness: yawning images were frequently predicted as no_yawn. The yawn recall is approximately 43%, so this class needs improvement.")
        st.markdown("#### Reported class-level results (MobileNetV2)")
        report = pd.DataFrame({
            "Class": ["Closed","Open","no_yawn","yawn"],
            "Precision": [1.00,1.00,0.63,0.96],
            "Recall": [1.00,1.00,0.98,0.43],
            "F1-score": [1.00,1.00,0.77,0.59],
            "Support": [109,109,108,109],
        })
        render_html_table(report)
    else:
        st.info("Model comparison CSV not found.")

elif page == "Fatigue Analysis":
    st.subheader("Fatigue progression (demonstration)")
    st.warning("This progression was calculated from shuffled test-image predictions—not timestamped driving video. The x-axis represents simulated intervals, not real minutes of driving.")
    st.markdown("### Mean fatigue score by interval")
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(progression["Minute"], progression["Mean fatigue score"], marker="o")
    ax.set_xticks(progression["Minute"])
    ax.set_ylim(0,2)
    ax.set_yticks([0,1,2], ["Alert","Mild Fatigue","Severe Fatigue"])
    ax.set_xlabel("Simulated interval")
    ax.set_ylabel("Mean fatigue score")
    ax.grid(True, alpha=.3)
    st.pyplot(fig, clear_figure=True)

    st.markdown("### Stage counts by interval")
    stage_cols = ["Alert frames", "Mild frames", "Severe frames"]
    fig, ax = plt.subplots(figsize=(10,4))
    x = progression["Minute"]
    bottom = np.zeros(len(progression))
    for col, label in zip(stage_cols, ["Alert", "Mild Fatigue", "Severe Fatigue"]):
        values = progression[col].to_numpy()
        ax.bar(x, values, bottom=bottom, label=label)
        bottom += values
    ax.set_xlabel("Simulated interval")
    ax.set_ylabel("Number of image predictions")
    ax.set_title("Fatigue stage counts by interval")
    ax.legend()
    st.pyplot(fig, clear_figure=True)
    st.markdown("### Progression data")
    render_html_table(progression)
    st.download_button("Download fatigue progression CSV", progression.to_csv(index=False), "fatigue_progression_results.csv", "text/csv")

elif page == "About":
    st.subheader("About this project")
    st.markdown("""
**Goal:** Demonstrate image classification for eye state and yawning, then map predictions to fatigue categories.

**Image classes**
- `Closed` — eyes closed
- `Open` — eyes open
- `no_yawn` — no yawn
- `yawn` — yawning

**Fatigue mapping used in the notebook**
- `Open` and `no_yawn` → Alert
- `yawn` → Mild Fatigue
- `Closed` → Severe Fatigue

**Models compared**
- Custom CNN trained from scratch
- MobileNetV2 using ImageNet transfer learning with its feature extractor frozen

**Limitations**
- A single image is not sufficient to infer a driver's actual fatigue level.
- The progression chart is simulated from test images, not continuous video.
- The model may be affected by lighting, camera angle, glasses, occlusion, and dataset bias.
- This app is an educational demonstration, not a medical, automotive, or road-safety product.
""")
