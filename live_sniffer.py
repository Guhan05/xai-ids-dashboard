import time
import json
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from scapy.all import sniff, IP, TCP, UDP

# 1. Load Trained Artifacts
print("Loading model artifacts...")
dnn_model = tf.keras.models.load_model("dashboard_dnn.keras")
ae_model = tf.keras.models.load_model("dashboard_ae.keras")
threshold = joblib.load("dashboard_threshold.pkl")

# Active Flow Cache: {flow_key: [packet_timestamps, packet_lengths]}
flow_cache = {}
INCIDENT_FILE = "live_incidents.json"

def process_packet(packet):
    if not packet.haslayer(IP):
        return

    src_ip = packet[IP].src
    dst_ip = packet[IP].dst
    protocol = "TCP" if packet.haslayer(TCP) else ("UDP" if packet.haslayer(UDP) else "OTHER")
    src_port = packet.sport if hasattr(packet, 'sport') else 0
    dst_port = packet.dport if hasattr(packet, 'dport') else 0
    pkt_len = len(packet)

    flow_key = (src_ip, dst_ip, src_port, dst_port, protocol)
    now = time.time()

    if flow_key not in flow_cache:
        flow_cache[flow_key] = {"start_time": now, "packets": [], "lengths": []}

    flow = flow_cache[flow_key]
    flow["packets"].append(now)
    flow["lengths"].append(pkt_len)

    # Evaluate flow every 10 packets or after 2 seconds
    if len(flow["packets"]) >= 10 or (now - flow["start_time"]) > 2.0:
        duration = now - flow["start_time"] if (now - flow["start_time"]) > 0 else 0.001
        total_bytes = sum(flow["lengths"])
        pkt_count = len(flow["packets"])
        
        # Extract live features matching model input dimensions (78 features)
        # Here we map basic flow statistics into a 78-length vector
        features = np.zeros((1, 78))
        features[0, 0] = dst_port                        # Destination Port
        features[0, 1] = duration                        # Flow Duration
        features[0, 2] = pkt_count                       # Total Fwd Packets
        features[0, 4] = total_bytes                     # Total Length Fwd Packets
        features[0, 8] = np.mean(flow["lengths"])         # Fwd Packet Length Mean
        features[0, 14] = total_bytes / duration         # Flow Bytes/s
        features[0, 15] = pkt_count / duration           # Flow Packets/s

        # 2. Run Inference
        recon_err = float(np.mean(np.square(features - ae_model.predict(features, verbose=0))))
        dnn_prob = float(dnn_model.predict(features, verbose=0).ravel()[0])

        if dnn_prob > 0.5:
            verdict = "Known Attack"
        elif recon_err > threshold:
            verdict = "Zero-Day Candidate"
        else:
            verdict = "Benign"

        # 3. Log Suspicious Activity
        if verdict != "Benign":
            incident = {
                "timestamp": time.strftime("%H:%M:%S"),
                "src_ip": src_ip,
                "src_port": src_port,
                "dst_ip": dst_ip,
                "dst_port": dst_port,
                "protocol": protocol,
                "verdict": verdict,
                "recon_err": recon_err,
                "dnn_prob": dnn_prob,
                "top_feature": "Flow Packets/s" if features[0, 15] > 100 else "Flow Duration"
            }
            
            # Append to shared JSON file for Streamlit to display
            incidents = []
            if os.path.exists(INCIDENT_FILE):
                try:
                    with open(INCIDENT_FILE, "r") as f:
                        incidents = json.load(f)
                except Exception:
                    incidents = []
            
            incidents.insert(0, incident)
            incidents = incidents[:100]  # Keep last 100 alerts
            
            with open(INCIDENT_FILE, "w") as f:
                json.dump(incidents, f, indent=2)

            print(f"🚨 [ALERT LOGGED] {verdict} from {src_ip}:{src_port} -> {dst_ip}:{dst_port}")

        # Reset flow cache entry
        del flow_cache[flow_key]

print("📡 Starting Live Packet Capture on Network Interface... Press Ctrl+C to stop.")
sniff(prn=process_packet, store=False)
