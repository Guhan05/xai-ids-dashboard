import os
import time
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import streamlit as st

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="🛡️ Real-Time XAI Threat Operations Center",
    page_icon="🛡️",
    layout="wide"
)

# ---------------------------------------------------------
# Initialize Persistent Session State
# ---------------------------------------------------------
if "incident_log" not in st.session_state:
    st.session_state.incident_log = []

if "blocked_ips" not in st.session_state:
    st.session_state.blocked_ips = set()

if "streaming" not in st.session_state:
    st.session_state.streaming = False

INCIDENT_FILE = "live_incidents.json"

# ---------------------------------------------------------
# Complete CICIDS2017 Feature Mapping Dictionary (78 Features)
# ---------------------------------------------------------
FEATURE_MAP = {
    "Feature_0": "Destination Port", "Feature_1": "Flow Duration", "Feature_2": "Total Fwd Packets",
    "Feature_3": "Total Backward Packets", "Feature_4": "Total Length of Fwd Packets", "Feature_5": "Total Length of Bwd Packets",
    "Feature_6": "Fwd Packet Length Max", "Feature_7": "Fwd Packet Length Min", "Feature_8": "Fwd Packet Length Mean",
    "Feature_9": "Fwd Packet Length Std", "Feature_10": "Bwd Packet Length Max", "Feature_11": "Bwd Packet Length Min",
    "Feature_12": "Bwd Packet Length Mean", "Feature_13": "Bwd Packet Length Std", "Feature_14": "Flow Bytes/s",
    "Feature_15": "Flow Packets/s", "Feature_16": "Flow IAT Mean", "Feature_17": "Flow IAT Std",
    "Feature_18": "Flow IAT Max", "Feature_19": "Flow IAT Min", "Feature_20": "Fwd IAT Total",
    "Feature_21": "Fwd IAT Mean", "Feature_22": "Fwd IAT Std", "Feature_23": "Fwd IAT Max",
    "Feature_24": "Fwd IAT Min", "Feature_25": "Bwd IAT Total", "Feature_26": "Bwd IAT Mean",
    "Feature_27": "Bwd IAT Std", "Feature_28": "Bwd IAT Max", "Feature_29": "Bwd IAT Min",
    "Feature_30": "Fwd PSH Flags", "Feature_31": "Bwd PSH Flags", "Feature_32": "Fwd URG Flags",
    "Feature_33": "Bwd URG Flags", "Feature_34": "Fwd Header Length", "Feature_35": "Bwd Header Length",
    "Feature_36": "Fwd Packets/s", "Feature_37": "Bwd Packets/s", "Feature_38": "Min Packet Length",
    "Feature_39": "Max Packet Length", "Feature_40": "Packet Length Mean", "Feature_41": "Packet Length Std",
    "Feature_42": "Packet Length Variance", "Feature_43": "FIN Flag Count", "Feature_44": "SYN Flag Count",
    "Feature_45": "RST Flag Count", "Feature_46": "PSH Flag Count", "Feature_47": "ACK Flag Count",
    "Feature_48": "URG Flag Count", "Feature_49": "CWE Flag Count", "Feature_50": "ECE Flag Count",
    "Feature_51": "Down/Up Ratio", "Feature_52": "Average Packet Size", "Feature_53": "Avg Fwd Segment Size",
    "Feature_54": "Avg Bwd Segment Size", "Feature_55": "Fwd Header Length.1", "Feature_56": "Fwd Avg Bytes/Bulk",
    "Feature_57": "Fwd Avg Packets/Bulk", "Feature_58": "Fwd Avg Bulk Rate", "Feature_59": "Bwd Avg Bytes/Bulk",
    "Feature_60": "Bwd Avg Packets/Bulk", "Feature_61": "Bwd Avg Bulk Rate", "Feature_62": "Subflow Fwd Packets",
    "Feature_63": "Subflow Fwd Bytes", "Feature_64": "Subflow Bwd Packets", "Feature_65": "Subflow Bwd Bytes",
    "Feature_66": "Init_Win_bytes_forward", "Feature_67": "Init_Win_bytes_backward", "Feature_68": "act_data_pkt_fwd",
    "Feature_69": "min_seg_size_forward", "Feature_70": "Active Mean", "Feature_71": "Active Std",
    "Feature_72": "Active Max", "Feature_73": "Active Min", "Feature_74": "Idle Mean",
    "Feature_75": "Idle Std", "Feature_76": "Idle Max", "Feature_77": "Idle Min"
}

def get_readable_feature(feature_key):
    clean_key = str(feature_key).strip()
    return FEATURE_MAP.get(clean_key, clean_key)

def generate_plain_english_explanation(verdict, top_features):
    readable_feats = [get_readable_feature(f) for f in top_features[:3]]
    feats_list_str = ", ".join([f"`{f}`" for f in readable_feats])
    
    if verdict == "Known Attack":
        return f"🚨 **Malicious Pattern Detected:** Flow matched known attack signature. Primary key indicators: {feats_list_str}."
    elif verdict == "Zero-Day Candidate":
        return f"⚠️ **Zero-Day Anomaly Detected:** High Autoencoder reconstruction error. Statistical anomalies in: {feats_list_str}."
    else:
        return "✅ **Normal Flow:** Traffic within expected operational baseline."

@st.cache_data
def generate_network_metadata(num_samples):
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
            st.error(f"Missing required artifact file: `{default_name}`.")
            st.stop()

    demo_data = np.load(resolved_paths["dashboard_demo_data.npz"])
    X_demo, y_demo, source_demo = demo_data["X"], demo_data["y"], demo_data["source"]

    dnn_model = tf.keras.models.load_model(resolved_paths["dashboard_dnn.keras"])
    ae_model = tf.keras.models.load_model(resolved_paths["dashboard_ae.keras"])
    threshold = joblib.load(resolved_paths["dashboard_threshold.pkl"])
    xai_df = pd.read_csv(resolved_paths["xai_consensus_results.csv"])

    return X_demo, y_demo, source_demo, dnn_model, ae_model, threshold, xai_df

X_demo, y_demo, source_demo, dnn_model, ae_model, threshold, xai_df = load_artifacts()
metadata_df = generate_network_metadata(len(X_demo))

# ---------------------------------------------------------
# Dashboard Header & Controls
# ---------------------------------------------------------
st.title("🛡️ Real-Time XAI Threat Operations Center")

st.sidebar.header("🕹️ Stream Engine Controls")
engine_source = st.sidebar.radio(
    "Data Input Mode",
    ["Local Packet Capture (live_sniffer.py)", "Demo Stream Simulator"]
)

stream_speed = st.sidebar.slider("Simulation Speed (s)", 0.2, 2.0, 0.5)

if st.sidebar.button("▶️ Start Stream Engine"):
    st.session_state.streaming = True

if st.sidebar.button("⏹️ Pause Stream"):
    st.session_state.streaming = False

if st.sidebar.button("🗑️ Clear Incident Logs"):
    st.session_state.incident_log = []
    st.session_state.blocked_ips = set()
    if os.path.exists(INCIDENT_FILE):
        os.remove(INCIDENT_FILE)
    st.rerun()

# Dynamic Containers
metrics_place = st.empty()
banner_place = st.empty()
table_place = st.empty()

# ---------------------------------------------------------
# UI Render Function
# ---------------------------------------------------------
def render_dashboard():
    # Sync live_incidents.json if local sniffer is running
    if os.path.exists(INCIDENT_FILE):
        try:
            with open(INCIDENT_FILE, "r") as f:
                live_file_data = json.load(f)
                for item in live_file_data:
                    if not any(i.get("timestamp") == item["timestamp"] and i.get("src_ip") == item["src_ip"] for i in st.session_state.incident_log):
                        item["status"] = "Pending Review"
                        item["shap_feats"] = [item.get("top_feature", "Feature_0")]
                        st.session_state.incident_log.insert(0, item)
        except Exception:
            pass

    # Top Metrics
    with metrics_place.container():
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Total Incidents Logged", len(st.session_state.incident_log))
        col_m2.metric("Active Blocked IPs", len(st.session_state.blocked_ips))
        zero_days = len([i for i in st.session_state.incident_log if i['verdict'] == 'Zero-Day Candidate'])
        col_m3.metric("Zero-Day Candidates", zero_days)
        attacks = len([i for i in st.session_state.incident_log if i['verdict'] == 'Known Attack'])
        col_m4.metric("Known Attack Alerts", attacks)
        st.divider()

    # Real-Time Incident Table
    with table_place.container():
        st.subheader("📋 Recorded Threat Incidents (Real-Time Stream)")
        if len(st.session_state.incident_log) == 0:
            st.info("No threats logged yet. Start `live_sniffer.py` or enable 'Demo Stream Simulator'.")
        else:
            df_log = pd.DataFrame(st.session_state.incident_log)
            cols = [c for c in ["timestamp", "flow_idx", "src_ip", "dst_ip", "dst_port", "verdict", "consensus_score", "top_feature", "status"] if c in df_log.columns]
            st.dataframe(df_log[cols], use_container_width=True)

render_dashboard()

# ---------------------------------------------------------
# LIVE STREAM PROCESSING LOOP (SIMULATION MODE)
# ---------------------------------------------------------
if st.session_state.streaming:
    if engine_source == "Demo Stream Simulator":
        idx = np.random.randint(0, len(X_demo))
        sample = X_demo[idx:idx+1]
        meta = metadata_df.iloc[idx].to_dict()

        recon_err = float(np.mean(np.square(sample - ae_model.predict(sample, verbose=0))))
        dnn_prob = float(dnn_model.predict(sample, verbose=0).ravel()[0])

        if dnn_prob > 0.5:
            verdict = "Known Attack"
        elif recon_err > threshold:
            verdict = "Zero-Day Candidate"
        else:
            verdict = "Benign"

        if verdict != "Benign":
            matching_xai = xai_df[xai_df["sample_id"] == idx]
            if not matching_xai.empty:
                raw_shap = matching_xai.iloc[0]["top_shap_features"]
                shap_feats = eval(raw_shap) if isinstance(raw_shap, str) else raw_shap
                score = float(matching_xai.iloc[0]["consensus_score"])
                action = matching_xai.iloc[0]["response_action"]
            else:
                top_dims = np.argsort(np.abs(sample[0]))[::-1][:3]
                shap_feats = [f"Feature_{d}" for d in top_dims]
                score = 0.8120
                action = "AUTOMATED_MITIGATION"

            if action == "AUTOMATED_MITIGATION":
                st.session_state.blocked_ips.add(meta['src_ip'])
                status = "Auto-Blocked"
            else:
                status = "Pending Review"

            incident_record = {
                "timestamp": time.strftime("%H:%M:%S"),
                "flow_idx": idx,
                "src_ip": meta['src_ip'],
                "src_port": meta['src_port'],
                "dst_ip": meta['dst_ip'],
                "dst_port": meta['dst_port'],
                "protocol": meta['protocol'],
                "verdict": verdict,
                "consensus_score": score,
                "top_feature": get_readable_feature(shap_feats[0]),
                "status": status,
                "shap_feats": shap_feats
            }
            
            st.session_state.incident_log.insert(0, incident_record)

        with banner_place.container():
            if verdict == "Benign":
                st.caption(f"📡 Processing Flow #{idx} — Status: Normal ({meta['src_ip']} ➔ {meta['dst_ip']})")
            else:
                st.error(f"🚨 ALERT! Flow #{idx} [{verdict}] from {meta['src_ip']} — Added to Live Table.")

    else: # Local Packet Capture Mode
        with banner_place.container():
            st.caption("📡 Listening for live packets via `live_incidents.json` from `live_sniffer.py`...")

    render_dashboard()
    time.sleep(stream_speed)
    st.rerun()

# ---------------------------------------------------------
# INTERACTIVE ANALYST INVESTIGATION PANEL
# ---------------------------------------------------------
st.divider()
st.subheader("🔍 Incident Investigation & Response Panel")

if len(st.session_state.incident_log) > 0:
    incident_options = [
        f"Flow #{i.get('flow_idx', 0)} - {i['verdict']} ({i['src_ip']}) @ {i['timestamp']}"
        for i in st.session_state.incident_log
    ]
    selected_option = st.selectbox("Select Incident to Investigate:", incident_options)
    
    selected_ts = selected_option.split(" @ ")[-1]
    inc = next(item for item in st.session_state.incident_log if item["timestamp"] == selected_ts)

    st.markdown(f"### Incident Details: {inc['verdict']} ({inc['src_ip']})")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Source IP", inc.get("src_ip", "Unknown"))
    c2.metric("Source Port", inc.get("src_port", "N/A"))
    c3.metric("Destination IP", inc.get("dst_ip", "Unknown"))
    c4.metric("Destination Port", inc.get("dst_port", "N/A"))
    c5.metric("Protocol", inc.get("protocol", "TCP"))

    st.markdown("#### 💬 XAI Diagnosis")
    shap_f = inc.get("shap_feats", [inc.get("top_feature", "Feature_0")])
    st.warning(generate_plain_english_explanation(inc["verdict"], shap_f))

    col_l, col_r = st.columns(2)
    with col_l:
        st.write(f"**XAI Consensus Score:** `{inc.get('consensus_score', 0.85):.4f}`")
        st.write(f"**Primary Driver:** `{inc.get('top_feature', 'Unknown')}`")
        st.write(f"**Status:** `{inc.get('status', 'Pending Review')}`")

    with col_r:
        st.markdown("#### ⚙️ Manual Actions")
        b1, b2, b3 = st.columns(3)
        if b1.button("🛡️ Block IP"):
            st.session_state.blocked_ips.add(inc["src_ip"])
            inc["status"] = "Analyst-Blocked"
            st.rerun()
        if b2.button("🟡 Quarantine"):
            inc["status"] = "Quarantined"
            st.rerun()
        if b3.button("✅ Dismiss"):
            inc["status"] = "Dismissed"
            st.rerun()
