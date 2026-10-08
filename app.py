import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="XAI Multi-Stage IDS Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# Title & Description
st.title("🛡️ XAI-Driven Multi-Stage Intrusion Detection System")
st.markdown("""
This dashboard monitors network traffic flows using a **Cascade Architecture**:
1. **Autoencoder (AE):** Detects zero-day anomaly candidates based on reconstruction error cutoffs.
2. **Deep Neural Network (DNN):** Classifies known attack vectors.
3. **Dual-Engine XAI (SHAP + LIME):** Evaluates explanation consensus to enforce safe, automated response gating.
""")

st.divider()

# ---------------------------------------------------------
# 1. Load Data and Artifacts
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    # Updated paths to point to the root directory
    data_path = "dashboard_demo_data.npz"
    dnn_path = "dashboard_dnn.keras"
    ae_path = "dashboard_ae.keras"
    thresh_path = "dashboard_threshold.pkl"
    csv_path = "xai_consensus_results.csv"

    # Verify asset existence
    required_files = [data_path, dnn_path, ae_path, thresh_path, csv_path]
    for f in required_files:
        if not os.path.exists(f):
            st.error(f"Missing required artifact: `{f}`.")
            st.stop()

    demo_data = np.load(data_path)
    X_demo = demo_data["X"]
    y_demo = demo_data["y"]
    source_demo = demo_data["source"]

    dnn_model = tf.keras.models.load_model(dnn_path)
    ae_model = tf.keras.models.load_model(ae_path)
    threshold = joblib.load(thresh_path)
    xai_df = pd.read_csv(csv_path)

    return X_demo, y_demo, source_demo, dnn_model, ae_model, threshold, xai_df
    
X_demo, y_demo, source_demo, dnn_model, ae_model, threshold, xai_df = load_artifacts()

# Compute predictions for all demo samples
recon_errors = np.mean(np.square(X_demo - ae_model.predict(X_demo, verbose=0)), axis=1)
dnn_probs = dnn_model.predict(X_demo, verbose=0).ravel()

verdicts = np.full(X_demo.shape[0], "Benign", dtype=object)
verdicts[dnn_probs > 0.5] = "Known Attack"
verdicts[(recon_errors > threshold) & (dnn_probs <= 0.5)] = "Zero-Day Candidate"

# ---------------------------------------------------------
# 2. Executive Metrics Panel
# ---------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

total_flows = len(X_demo)
benign_cnt = np.sum(verdicts == "Benign")
known_cnt = np.sum(verdicts == "Known Attack")
zero_day_cnt = np.sum(verdicts == "Zero-Day Candidate")

col1.metric("Total Analyzed Flows", f"{total_flows:,}")
col2.metric("Benign Flows", f"{benign_cnt:,}", delta="Normal", delta_color="normal")
col3.metric("Known Attacks (DNN)", f"{known_cnt:,}", delta="Alert", delta_color="inverse")
col4.metric("Zero-Day Candidates (AE)", f"{zero_day_cnt:,}", delta="Critical", delta_color="inverse")

st.divider()

# ---------------------------------------------------------
# 3. Anomaly Threshold & Traffic Overview Plots
# ---------------------------------------------------------
c_left, c_right = st.columns([1, 1])

with c_left:
    st.subheader("📊 Traffic Verdict Distribution")
    verdict_df = pd.DataFrame({"Verdict": verdicts}).value_counts().reset_index()
    verdict_df.columns = ["Verdict", "Count"]
    
    fig_pie = px.pie(
        verdict_df, 
        names="Verdict", 
        values="Count", 
        color="Verdict",
        color_discrete_map={
            "Benign": "#2ecc71",
            "Known Attack": "#e74c3c",
            "Zero-Day Candidate": "#f39c12"
        },
        hole=0.4
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with c_right:
    st.subheader("📉 Autoencoder Reconstruction Error vs. Cutoff")
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        y=recon_errors,
        mode='markers',
        marker=dict(color=np.where(recon_errors > threshold, '#e74c3c', '#3498db'), size=6),
        name="Flow Error"
    ))
    fig_scatter.add_hline(
        y=threshold, 
        line_dash="dash", 
        line_color="orange", 
        annotation_text=f"Cutoff Threshold: {threshold:.4f}",
        annotation_position="bottom right"
    )
    fig_scatter.update_layout(
        xaxis_title="Flow Sample Index",
        yaxis_title="MSE Reconstruction Error",
        template="plotly_white"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

st.divider()

# ---------------------------------------------------------
# 4. Phase 5: XAI Consensus & Gating Inspector
# ---------------------------------------------------------
st.subheader("🔍 Dual-Engine XAI Consensus & Gating Inspector")

if xai_df.empty:
    st.info("No flagged XAI evaluation records found.")
else:
    sample_ids = xai_df["sample_id"].tolist()
    selected_id = st.selectbox("Select Flagged Sample ID for Analysis:", sample_ids)

    # Filter data for selected sample
    sample_row = xai_df[xai_df["sample_id"] == selected_id].iloc[0]

    meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)
    meta_col1.markdown(f"**Source Group:** `{sample_row['source']}`")
    meta_col2.markdown(f"**Cascade Verdict:** `{sample_row['verdict']}`")
    meta_col3.markdown(f"**Consensus Score:** `{sample_row['consensus_score']:.4f}`")

    # Dynamic Gating Badge
    action = sample_row["response_action"]
    if action == "AUTOMATED_MITIGATION":
        meta_col4.error("🛡️ Action: AUTOMATED MITIGATION")
    else:
        meta_col4.warning("⚠️ Action: HUMAN ANALYST REVIEW")

    st.markdown("### Top Feature Attributions (SHAP vs. LIME)")
    
    # Format feature strings safely
    shap_feats = sample_row["top_shap_features"]
    lime_feats = sample_row["top_lime_features"]
    
    if isinstance(shap_feats, str):
        shap_feats = eval(shap_feats)
    if isinstance(lime_feats, str):
        lime_feats = eval(lime_feats)

    f_col1, f_col2 = st.columns(2)
    
    with f_col1:
        st.markdown("**SHAP (Global/Sampling-based) Top Features:**")
        for i, feat in enumerate(shap_feats, 1):
            st.markdown(f"{i}. `{feat}`")

    with f_col2:
        st.markdown("**LIME (Local Linear Surrogate) Top Features:**")
        for i, feat in enumerate(lime_feats, 1):
            st.markdown(f"{i}. `{feat}`")

    # Consensus Gauge Chart
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=sample_row["consensus_score"],
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "SHAP-LIME Explanation Consensus Score"},
        gauge={
            'axis': {'range': [0, 1]},
            'bar': {'color': "#2980b9"},
            'steps': [
                {'range': [0, 0.70], 'color': "#f1c40f"},
                {'range': [0.70, 1.0], 'color': "#2ecc71"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.70
            }
        }
    ))
    fig_gauge.update_layout(height=300)
    st.plotly_chart(fig_gauge, use_container_width=True)
