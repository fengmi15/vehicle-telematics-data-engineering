import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.feature_engineering import compute_driving_features

st.set_page_config(page_title="Vehicle Telematics Data Engineering", layout="wide", page_icon="🚗")

st.title("🚗 Vehicle Telematics — Data Engineering Pipeline")
st.caption("CAN/OBD-II simulation -> realistic noise injection -> cleaning pipeline -> recovery analysis")

clean   = pd.read_csv("data/clean_drive.csv")
noisy   = pd.read_csv("data/decoded_noisy_drive.csv")
cleaned = pd.read_csv("data/cleaned_drive.csv")

results = {
    "Clean (Ground Truth)": compute_driving_features(clean),
    "Noisy (Raw CAN)":      compute_driving_features(noisy),
    "Cleaned (Recovered)":  compute_driving_features(cleaned),
}

# ── Top metrics ──
col1, col2, col3 = st.columns(3)
for col, (label, feats) in zip([col1,col2,col3], results.items()):
    with col:
        st.subheader(label)
        st.metric("Driving Style Score", feats["driving_style_score"])
        st.metric("Engine Stress Index", feats["engine_stress_index"])
        st.metric("Harsh Braking Events", feats["harsh_braking_events"])
        st.metric("Sample Count", feats["sample_count"])

st.divider()

# ── Signal comparison chart ──
st.subheader("Signal Comparison (first 600 samples)")
fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                     subplot_titles=["Speed (km/h)", "RPM", "Coolant Temp (C)"])

for i, col in enumerate(["speed_kmh","rpm","coolant_temp"], start=1):
    fig.add_trace(go.Scatter(y=clean[col][:600], name="Clean", line=dict(color="#22c55e")), row=i, col=1)
    fig.add_trace(go.Scatter(y=noisy[col][:600], name="Noisy", line=dict(color="#ef4444", width=1), opacity=0.5), row=i, col=1)
    fig.add_trace(go.Scatter(y=cleaned[col][:600], name="Cleaned", line=dict(color="#3b82f6")), row=i, col=1)

fig.update_layout(height=700, plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
                   font=dict(color="#e2e8f0"), showlegend=True)
fig.update_xaxes(gridcolor="#1e293b")
fig.update_yaxes(gridcolor="#1e293b")
st.plotly_chart(fig, use_container_width=True)

# ── Recovery table ──
st.subheader("Feature Recovery Accuracy")
import numpy as np
rows = []
for key, gt_val in results["Clean (Ground Truth)"].items():
    if key == "sample_count": continue
    noisy_val = results["Noisy (Raw CAN)"][key]
    clean_val = results["Cleaned (Recovered)"][key]
    if gt_val == 0: continue
    recovery = round(100 * (1 - abs(gt_val - clean_val) / max(abs(gt_val), 1e-6)), 1)
    rows.append({"Metric": key, "Ground Truth": gt_val, "Noisy": noisy_val,
                  "Cleaned": clean_val, "Recovery %": recovery})

st.dataframe(pd.DataFrame(rows), use_container_width=True)