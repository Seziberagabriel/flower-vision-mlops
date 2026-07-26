"""
ui/app.py
Streamlit dashboard for the Flower Vision MLOps pipeline.
"""

import os
import requests
import streamlit as st
import matplotlib.pyplot as plt

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Flower Vision MLOps", page_icon="🌼", layout="wide")
st.title("🌼 Flower Vision — MLOps Dashboard")


def bar_chart(data: dict, ylabel: str = ""):
    if not data:
        st.info("No data to display yet.")
        return
    fig, ax = plt.subplots(figsize=(5, 3))
    labels = list(data.keys())
    values = [v if v is not None else 0 for v in data.values()]
    ax.bar(labels, values, color="#4C9A6B")
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


with st.sidebar:
    st.header("📡 Model / API Status")
    try:
        resp = requests.get(f"{API_URL}/status", timeout=5)
        data = resp.json()
        st.metric("Status", data.get("status", "unknown"))
        uptime_min = round(data.get("uptime_seconds", 0) / 60, 1)
        st.metric("API Uptime", f"{uptime_min} min")
        st.metric("Model Accuracy", f"{data.get('model_accuracy', 0):.2%}" if data.get("model_accuracy") else "N/A")
        st.write("Last retrained:", data.get("model_last_trained") or "never")
        st.write("Pending uploaded images:", data.get("pending_uploaded_images", 0))
        job = data.get("retrain_job", {})
        st.write("Retrain job status:", job.get("status"))
    except Exception as e:
        st.error(f"Could not reach API at {API_URL}: {e}")

tab1, tab2, tab3 = st.tabs(["Predict", "Data Visualizations", "Bulk Upload & Retrain"])

with tab1:
    st.subheader("Predict the species of a single flower photo")
    uploaded_file = st.file_uploader("Upload a flower image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded image", width=300)
        if st.button("Predict"):
            with st.spinner("Calling the model..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                r = requests.post(f"{API_URL}/predict", files=files, timeout=30)
            if r.status_code == 200:
                result = r.json()
                st.success(f"Prediction: **{result['predicted_class']}** "
                           f"({result['confidence']:.1%} confidence)")
                bar_chart(result["all_probabilities"], ylabel="Probability")
            else:
                st.error(f"Error: {r.json().get('detail', r.text)}")

with tab2:
    st.subheader("Dataset insights")
    try:
        viz = requests.get(f"{API_URL}/visualizations", timeout=10).json()

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Class distribution** (feature 1: class balance)")
            dist = viz.get("class_distribution", {})
            if dist:
                bar_chart(dist, ylabel="Number of images")
                st.caption(
                    "Story: classes are roughly balanced, so accuracy is a fair headline "
                    "metric here — no single class dominates the training set."
                )

        with col2:
            st.write("**Average brightness per class** (feature 2: pixel intensity)")
            brightness = viz.get("average_brightness_per_class", {})
            if brightness:
                bar_chart(brightness, ylabel="Mean pixel intensity (0-255)")
                st.caption(
                    "Story: some species (e.g. dandelions) tend to be photographed in "
                    "brighter/lighter conditions than others (e.g. roses) — useful context "
                    "if the model later struggles more on darker images."
                )

        st.write("**Model evaluation metrics** (feature 3: how good is the model)")
        metrics = viz.get("model_metrics", {})
        if metrics:
            m_cols = st.columns(4)
            m_cols[0].metric("Accuracy", f"{metrics.get('accuracy', 0):.2%}")
            m_cols[1].metric("Precision (macro)", f"{metrics.get('precision_macro', 0):.2%}")
            m_cols[2].metric("Recall (macro)", f"{metrics.get('recall_macro', 0):.2%}")
            m_cols[3].metric("F1 (macro)", f"{metrics.get('f1_macro', 0):.2%}")
            st.caption(
                "Story: comparing precision vs. recall per class in the notebook shows which "
                "flower species are most often confused with one another (see confusion matrix)."
            )
    except Exception as e:
        st.error(f"Could not load visualizations: {e}")

with tab3:
    st.subheader("Bulk upload new labeled images")
    st.caption(
        "Upload a **.zip** file structured as `class_name/*.jpg` "
        "(e.g. `daisy/img1.jpg`, `roses/img2.jpg`) to stage new training data."
    )
    zip_file = st.file_uploader("Upload ZIP of new images", type=["zip"])
    if zip_file is not None and st.button("Upload for retraining"):
        with st.spinner("Uploading..."):
            files = {"file": (zip_file.name, zip_file.getvalue(), "application/zip")}
            r = requests.post(f"{API_URL}/upload", files=files, timeout=60)
        if r.status_code == 200:
            st.success(r.json()["message"] + f" Total pending images: {r.json()['total_pending_images']}")
        else:
            st.error(f"Error: {r.json().get('detail', r.text)}")

    st.divider()
    st.subheader("Trigger retraining")
    st.caption(
        "Fine-tunes the **existing production model** on all uploaded-but-not-yet-trained "
        "images, evaluates it, and only promotes it if accuracy doesn't regress."
    )
    if st.button("Retrain model now"):
        with st.spinner("Kicking off retraining job..."):
            r = requests.post(f"{API_URL}/retrain", timeout=30)
        if r.status_code == 200:
            st.success(r.json()["message"])
            st.info("Refresh this page / check the sidebar to track job status.")
        else:
            st.error(f"Error: {r.json().get('detail', r.text)}")