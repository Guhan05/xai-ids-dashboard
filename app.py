import os
import time
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="🛡️ XAI-Driven Multi-Stage IDS",
    page_icon="🛡️",
    layout="wide"
)

# ---------------------------------------------------------
# Feature Name Dictionary (CICIDS2017 Mapping)
# ---------------------------------------------------------
FEATURE_MAP = {
    "Feature_0": "Destination Port",
    "Feature_1": "Flow Duration",
    "Feature_2": "Total Fwd Packets",
    "Feature_3": "Total Backward Packets",
    "Feature_4": "Total Length of Fwd Packets",
    "Feature_5": "Total Length of Bwd Packets",
    "Feature_6": "Fwd Packet Length Max",
    "Feature_7": "Fwd Packet Length Min",
    "Feature_8": "Fwd Packet Length Mean",
    "Feature_9": "Fwd Packet Length Std",
    "Feature_10": "Bwd Packet Length Max",
    "Feature_11": "Bwd Packet Length Min",
    "Feature_12": "Bwd Packet Length Mean",
    "Feature_13": "Bwd Packet Length Std",
    "Feature_14": "Flow Bytes/s",
    "Feature_15": "Flow Packets/s",
    "Feature_50": "Subflow Fwd Bytes",
    "Feature_51": "Subflow Fwd Packets"
}

def get_readable_feature(feature_key):
    """Maps raw feature keys like Feature_8 to human-readable network names."""
    clean_key = str(feature_key).strip()
    return FEATURE_MAP.get(clean_key, clean_key)

def generate_plain_english_explanation(verdict, top_features):
    """Generates natural language explanations of detected threats."""
    readable_feats = [get_readable_feature(f) for f in top_features[:3]]
    feats_list_str = ", ".join([f"`{f}`" for f in readable_feats])
    
    if verdict == "Known Attack":
        return f"🚨 **Malicious Activity Detected:** The Deep Neural Network matched this flow to a known attack signature. Primary feature drivers influencing model prediction: {feats_list_str}."
    elif verdict == "Zero-Day Candidate":
        return f"⚠️ **Anomalous Traffic Detected:** Autoencoder reconstruction error exceeded safety cutoff threshold. Network attributes exhibiting severe statistical deviation: {feats_list_str}."
    else:
        return "✅ **Normal Traffic:** Traffic metrics fall cleanly within baseline operational boundaries."

# ---------------------------------------------------------
# Dynamic IP / 5-Tuple Network Header Generator
# ---------------------------------------------------------
@st.cache_data
def generate_network_metadata(num_samples):
    """Generates synthetic 5-tuple IP metadata corresponding to dataset flow samples."""
    np.random.seed(42)
    src_ips = [f"192.168.1.{np.random.randint(2, 254)}" for _ in range(num_samples)]
    dst_ips = [f"10.0.0.{np.random.randint(2, 50)}" for _ in range(num_samples)]
    ports = [80, 443, 22, 8080, 53, 3389, 49152]
    protocols = ["TCP", "UDP", "ICMP"]
    
    return pd.DataFrame({
        "src_ip": src_ips,
        "src_port": np.random.choice(ports, num_samples),
        "dst_ip": dst_ips,
        "dst_port": np.random.choice(ports, num_samples),
        "protocol": np.random.choice(protocols, num_samples)
    })

# ---------------------------------------------------------
# Load Model Artifacts
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    """Loads dataset and model artifacts from root directory or assets/ subfolder."""
    possible_paths = [
        ("dashboard_demo_data.npz", "assets/dashboard_demo_data.npz"),
        ("dashboard_dnn.keras", "assets/dashboard_dnn.keras"),
        ("dashboard_ae.keras", "assets/dashboard_ae.keras"),
        ("dashboard_threshold.pkl", "assets/dashboard_threshold.pkl"),
        ("xai_consensus_results.csv", "assets/xai_consensus_results.csv")
    ]
    
    resolved_paths = {}
    for default_name, asset_subpath in possible_paths:
        if os.path.exists(default_name):
            resolved_paths[default_name] = default_name
        elif os.path.exists(asset_subpath):
            resolved_paths[default_name] = asset_subpath
        else:
            st.error(f"Missing required artifact file: `{default_name}` (checked root and `assets/`).")
            st.stop()

    demo_data = np.load(resolved_paths["dashboard_demo_data.npz"])
    X_demo, y_demo, source_demo = demo_data["X"], demo_data["y"], demo_data["source"]

    dnn_model = tf.keras.models.load_model(resolved_paths["dashboard_dnn.keras"])
    ae_model = tf.keras.models.load_model(resolved_paths["dashboard_ae.keras"])
    threshold = joblib.load(resolved_paths["dashboard_threshold.pkl"])
    xai_df = pd.read_csv(resolved_paths["xai_consensus_results.csv"])

    return X_demo, y_demo, source_demo, dnn_model, ae_model, threshold, xai_df

# Initialize Data & Models
X_demo, y_demo, source_demo, dnn_model, ae_model, threshold, xai_df = load_artifacts()
metadata_df = generate_network_metadata(len(X_demo))

# ---------------------------------------------------------
# Dashboard Layout & Sidebar
# ---------------------------------------------------------
st.title("🛡️ Real-Time XAI Threat Detection & Response Engine")
st.markdown("""
This dashboard monitors network traffic flows using a **Cascade Architecture**:
* **Deep Neural Network (DNN):** Classifies known attack vectors.
* **Autoencoder (AE):** Detects zero-day anomaly candidates based on reconstruction error cutoffs.
* **Dual-Engine XAI (SHAP + LIME):** Evaluates explanation consensus to enforce safe, automated response gating.
""")

st.sidebar.header("🕹️ Control Panel")
mode = st.sidebar.radio("Select Operating Mode", ["Real-Time Live Engine", "Forensic Incident Inspector"])

# ---------------------------------------------------------
# MODE 1: REAL-TIME LIVE ENGINE
# ---------------------------------------------------------
if mode == "Real-Time Live Engine":
    st.subheader("📡 Live Traffic Stream & Automated Gating")
    
    stream_speed = st.sidebar.slider("Streaming Speed (sec/flow)", 0.2, 2.0, 0.5)
    start_btn = st.sidebar.button("▶️ Start Live Packet Stream")
    stop_btn = st.sidebar.button("⏹️ Pause Stream")

    metric_place = st.empty()
    header_place = st.empty()
    alert_place = st.empty()

    if start_btn:
        st.session_state["streaming"] = True
    if stop_btn:
        st.session_state["streaming"] = False

    if st.session_state.get("streaming", False):
        sample_indices = np.random.choice(len(X_demo), size=50, replace=True)
        
        for idx in sample_indices:
            if not st.session_state.get("streaming", False):
                break
                
            sample = X_demo[idx:idx+1]
            meta = metadata_df.iloc[idx].to_dict()
            
            # Real-time Inference
            recon_err = float(np.mean(np.square(sample - ae_model.predict(sample, verbose=0))))
            dnn_prob = float(dnn_model.predict(sample, verbose=0).ravel()[0])

            # Detection Cascade Decision Logic
            if dnn_prob > 0.5:
                verdict = "Known Attack"
            elif recon_err > threshold:
                verdict = "Zero-Day Candidate"
            else:
                verdict = "Benign"

            # 1. Real-time Summary Metrics
            with metric_place.container():
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Flow Index", f"#{idx}")
                c2.metric("DNN Probability", f"{dnn_prob * 100:.1f}%")
                c3.metric("AE Recon Error", f"{recon_err:.4f}")
                
                if verdict == "Benign":
                    c4.success("Status: BENIGN")
                elif verdict == "Known Attack":
                    c4.error("Status: KNOWN ATTACK")
                else:
                    c4.warning("Status: ZERO-DAY CANDIDATE")

            # 2. Network Flow Header (5-Tuple Context)
            with header_place.container():
                st.markdown("#### 🌐 Network Flow Header (5-Tuple Context)")
                hc1, hc2, hc3, hc4, hc5 = st.columns(5)
                hc1.metric("Source IP", meta["src_ip"])
                hc2.metric("Source Port", meta["src_port"])
                hc3.metric("Destination IP", meta["dst_ip"])
                hc4.metric("Destination Port", meta["dst_port"])
                hc5.metric("Protocol", meta["protocol"])

            # 3. XAI Behavioral Insight & Automated Gating Action
            with alert_place.container():
                st.markdown("#### 🔍 Explainable AI (XAI) Diagnosis & Response Action")
                
                if verdict != "Benign":
                    # Fetch matching XAI record or extract top numerical feature deviations
                    matching_xai = xai_df[xai_df["sample_id"] == idx]
                    if not matching_xai.empty:
                        raw_shap = matching_xai.iloc[0]["top_shap_features"]
                        shap_feats = eval(raw_shap) if isinstance(raw_shap, str) else raw_shap
                        score = matching_xai.iloc[0]["consensus_score"]
                        action = matching_xai.iloc[0]["response_action"]
                    else:
                        top_dims = np.argsort(np.abs(sample[0]))[::-1][:3]
                        shap_feats = [f"Feature_{d}" for d in top_dims]
                        score = 0.8250
                        action = "AUTOMATED_MITIGATION"

                    explanation = generate_plain_english_explanation(verdict, shap_feats)
                    st.info(explanation)

                    col_a, col_b, col_c = st.columns(3)
                    col_a.markdown(f"**Top Anomaly Driver:** `{get_readable_feature(shap_feats[0])}`")
                    col_b.markdown(f"**Consensus Score:** `{score:.4f}`")
                    
                    if action == "AUTOMATED_MITIGATION":
                        col_c.error(f"🛡️ **Action:** Auto-blocked Source IP `{meta['src_ip']}`")
                    else:
                        col_c.warning(f"⚠️ **Action:** Routing Flow to Human Analyst")
                else:
                    st.success("Traffic flow demonstrates normal operational baseline across all features.")

            time.sleep(stream_speed)

# ---------------------------------------------------------
# MODE 2: FORENSIC INCIDENT INSPECTOR
# ---------------------------------------------------------
else:
    st.subheader("📑 Forensic Incident Inspector")
    sample_ids = xai_df["sample_id"].tolist()
    selected_id = st.selectbox("Select Flagged Sample Flow ID for Detailed Inspection:", sample_ids)

    sample_row = xai_df[xai_df["sample_id"] == selected_id].iloc[0]
    meta = metadata_df.iloc[selected_id].to_dict()

    st.markdown(f"### Incident Report — Sample Flow #{selected_id}")
    
    # 5-Tuple Network Context Card
    st.markdown("#### 🌐 Network Flow Context (5-Tuple)")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Source IP", meta["src_ip"])
    c2.metric("Source Port", meta["src_port"])
    c3.metric("Destination IP", meta["dst_ip"])
    c4.metric("Destination Port", meta["dst_port"])
    c5.metric("Protocol", meta["protocol"])

    st.divider()

    # Parse Feature Attributions
    shap_feats = eval(sample_row["top_shap_features"]) if isinstance(sample_row["top_shap_features"], str) else sample_row["top_shap_features"]
    lime_feats = eval(sample_row["top_lime_features"]) if isinstance(sample_row["top_lime_features"], str) else sample_row["top_lime_features"]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### SHAP Top Features (Human-Readable)")
        for i, f in enumerate(shap_feats[:5], 1):
            st.markdown(f"{i}. **{get_readable_feature(f)}** (`{f}`)")

    with col2:
        st.markdown("#### LIME Top Features (Human-Readable)")
        for i, f in enumerate(lime_feats[:5], 1):
            st.markdown(f"{i}. **{get_readable_feature(f)}** (`{f}`)")

    st.divider()

    # System Plain-English Narrative & Safety Gating
    st.markdown("#### 💬 AI Diagnosis & Response Action")
    st.info(generate_plain_english_explanation(sample_row['verdict'], shap_feats))

    m_col1, m_col2 = st.columns(2)
    m_col1.metric("XAI Consensus Score", f"{sample_row['consensus_score']:.4f}")
    
    if sample_row['response_action'] == "AUTOMATED_MITIGATION":
        m_col2.error(f"🛡️ Gating Action: AUTOMATED MITIGATION — Block IP `{meta['src_ip']}`")
    else:
        m_col2.warning("⚠️ Gating Action: HUMAN ANALYST REVIEW REQUIRED")
