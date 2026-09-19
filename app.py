"""
XAI-IDS — Live SOC Dashboard (Skeleton)
========================================
Simulates real-time traffic by replaying pre-processed flow records through
the fast detection path (autoencoder anomaly gate + DNN classifier) at a
controlled rate, showing per-flow latency and running verdict counts as if
watching a live feed.

This is deliberately a SKELETON: SHAP/LIME explanation, the consensus score,
and fuzzy criticality scoring (Phases 5-7) are not built yet, so flagged
instances show an "explanation pending" placeholder instead of a real
explanation. That placeholder is not a cosmetic stand-in — it's the actual
architecture: detection is synchronous (you'll see it happen instantly per
flow), explanation is asynchronous by design and will fill in this same slot
once Phase 5 exists, without changing this file's structure.

Run locally:   pip install streamlit tensorflow numpy pandas joblib
               streamlit run app.py
Or deploy free on Streamlit Community Cloud by pushing this file + the
four exported artifacts (dashboard_dnn.keras, dashboard_ae.keras,
dashboard_threshold.pkl, dashboard_demo_data.npz) to a GitHub repo.
"""

import time
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import tensorflow as tf

st.set_page_config(page_title="XAI-IDS — Live SOC Dashboard", layout="wide")

DATA_PATH = "dashboard_demo_data.npz"
DNN_PATH = "dashboard_dnn.keras"
AE_PATH = "dashboard_ae.keras"
THRESHOLD_PATH = "dashboard_threshold.pkl"


@st.cache_resource
def load_models():
    dnn = tf.keras.models.load_model(DNN_PATH)
    ae = tf.keras.models.load_model(AE_PATH)
    threshold = joblib.load(THRESHOLD_PATH)
    return dnn, ae, threshold


@st.cache_data
def load_demo_data():
    d = np.load(DATA_PATH, allow_pickle=True)
    return d["X"], d["y"], d["source"]


def reconstruction_error(model, X_row):
    recon = model.predict(X_row, verbose=0)
    return float(np.mean(np.square(X_row - recon)))


def classify_flow(ae, dnn, threshold, x_row):
    """Runs the synchronous fast path on a single flow. Returns verdict + timing."""
    x_row = x_row.reshape(1, -1)
    t0 = time.perf_counter()

    error = reconstruction_error(ae, x_row)
    is_anomalous = error > threshold

    proba = float(dnn.predict(x_row, verbose=0).ravel()[0])
    is_known_attack = proba > 0.5

    if is_known_attack:
        verdict = "known_attack"
    elif is_anomalous:
        verdict = "zero_day_candidate"
    else:
        verdict = "benign"

    latency_ms = (time.perf_counter() - t0) * 1000
    return verdict, error, proba, latency_ms


VERDICT_COLOR = {
    "benign": "🟢",
    "known_attack": "🟠",
    "zero_day_candidate": "🔴",
}


def explanation_placeholder(verdict):
    if verdict == "benign":
        return "—"
    return "⏳ Queued for SHAP/LIME (Phase 5)"


# ---------------- Sidebar controls ----------------
st.sidebar.header("Replay Controls")
n_flows = st.sidebar.slider("Flows to replay", min_value=10, max_value=347, value=100)
delay = st.sidebar.slider("Delay per flow (seconds)", min_value=0.0, max_value=1.0, value=0.15, step=0.05)
start = st.sidebar.button("▶ Start Replay", type="primary")
st.sidebar.caption(
    "Simulates a live traffic stream by replaying stored flows at a controlled rate. "
    "Detection (autoencoder + DNN) runs synchronously per flow, matching the fast-path "
    "design — explanation generation is intentionally deferred to a later phase."
)

st.title("🛡️ XAI-IDS — Live SOC Dashboard")
st.caption("Fast-path detection running in simulated real time. Explanation layer: not yet built (Phase 5).")

try:
    dnn_model, ae_model, threshold = load_models()
    X_demo, y_demo, source_demo = load_demo_data()
    models_ok = True
except Exception as e:
    models_ok = False
    st.error(
        "Couldn't load model/data files. Make sure dashboard_dnn.keras, dashboard_ae.keras, "
        "dashboard_threshold.pkl, and dashboard_demo_data.npz are in the same folder as this "
        f"script.\n\nDetails: {e}"
    )

if models_ok:
    metric_cols = st.columns(4)
    benign_metric = metric_cols[0].empty()
    known_metric = metric_cols[1].empty()
    zeroday_metric = metric_cols[2].empty()
    latency_metric = metric_cols[3].empty()

    table_placeholder = st.empty()
    summary_placeholder = st.container()

    if start:
        log = []
        n_flows = min(n_flows, len(X_demo))

        for i in range(n_flows):
            x_row = X_demo[i]
            true_label = "attack" if y_demo[i] == 1 else "benign"
            source = source_demo[i]

            verdict, error, proba, latency_ms = classify_flow(ae_model, dnn_model, threshold, x_row)

            log.append({
                "Flow #": i + 1,
                "Source": source,
                "Verdict": f"{VERDICT_COLOR[verdict]} {verdict}",
                "Recon. Error": round(error, 5),
                "DNN P(attack)": round(proba, 4),
                "Latency (ms)": round(latency_ms, 2),
                "Explanation": explanation_placeholder(verdict),
                "Ground Truth": true_label,
            })

            log_df = pd.DataFrame(log)
            counts = log_df["Verdict"].str.contains
            benign_metric.metric("🟢 Benign", int(counts("benign").sum()))
            known_metric.metric("🟠 Known Attack", int(counts("known_attack").sum()))
            zeroday_metric.metric("🔴 Zero-Day Candidate", int(counts("zero_day_candidate").sum()))
            latency_metric.metric("Avg Latency", f"{log_df['Latency (ms)'].mean():.2f} ms")

            table_placeholder.dataframe(
                log_df.iloc[::-1].head(20), use_container_width=True, hide_index=True
            )
            time.sleep(delay)

        with summary_placeholder:
            st.divider()
            st.subheader("Replay Summary")
            zd_rows = log_df[log_df["Source"] == "zero_day_holdout"]
            if len(zd_rows) > 0:
                caught = zd_rows["Verdict"].str.contains("zero_day_candidate").sum() + \
                         zd_rows["Verdict"].str.contains("known_attack").sum()
                st.write(
                    f"Of {len(zd_rows)} true zero-day flows replayed, **{caught} "
                    f"({100*caught/len(zd_rows):.1f}%)** were flagged by some stage of the cascade."
                )
            st.write(f"Average end-to-end detection latency: **{log_df['Latency (ms)'].mean():.2f} ms/flow**")
    else:
        st.info("Set your replay parameters in the sidebar, then click **Start Replay**.")
