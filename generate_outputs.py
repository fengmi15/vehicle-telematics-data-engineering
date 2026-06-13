import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

os.makedirs("outputs", exist_ok=True)

clean   = pd.read_csv("data/clean_drive.csv")
noisy   = pd.read_csv("data/decoded_noisy_drive.csv").dropna().reset_index(drop=True) #Line added to remove spikes 
cleaned = pd.read_csv("data/cleaned_drive.csv").dropna().reset_index(drop=True) #line added to remove spikes 

print(f"Clean timestamps (first 5): {clean['timestamp'].head().tolist()}")
print(f"Cleaned timestamps (first 5): {cleaned['timestamp'].head().tolist()}")
print(f"Clean dtype: {clean['timestamp'].dtype}, Cleaned dtype: {cleaned['timestamp'].dtype}")

# ── CREATE t_sec COLUMNS ──
clean["t_sec"]   = (pd.to_numeric(clean["timestamp"]) // 1000).astype(int)
cleaned["t_sec"] = (pd.to_numeric(cleaned["timestamp"]) // 1000).astype(int)

# Check if they overlap at all
clean_secs = set(clean["t_sec"])
cleaned_secs = set(cleaned["t_sec"])
print(f"Overlap: {len(clean_secs & cleaned_secs)} / {len(clean_secs)} samples")

merged = pd.merge(clean, cleaned, on="t_sec", suffixes=("_clean","_cleaned"), how="inner")
merged = merged.iloc[:600]

# ── Plot 1: 3-panel signal comparison (speed, rpm, coolant) ──
fig, axes = plt.subplots(3, 1, figsize=(14, 10), facecolor="white", sharex=False)
fig.suptitle("CAN/OBD Signal: Clean vs Noisy vs Cleaned", color="#1e293b", fontsize=14, y=0.98)
signals = [("speed_kmh","Speed (km/h)"), ("rpm","RPM"), ("coolant_temp","Coolant Temp (C)")]

for ax, (col, label) in zip(axes, signals):
    ax.set_facecolor("white")
    ax.plot(merged[f"{col}_clean"].values[:600], color="#22c55e", linewidth=1.5, label="Clean (ground truth)")
    ax.plot(noisy[col].values[:600], color="#ef4444", linewidth=1.0, alpha=0.6, label="Noisy (raw)")
    ax.plot(merged[f"{col}_cleaned"].values[:600], color="#3b82f6", linewidth=1.5, label="Cleaned (recovered)")
    ax.set_ylabel(label, color="#334155", fontsize=9)
    ax.tick_params(colors="#475569", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")
    ax.grid(axis="y", color="#e2e8f0", linewidth=0.8)
    ax.legend(facecolor="white", edgecolor="#cbd5e1", labelcolor="#1e293b", fontsize=8)

axes[-1].set_xlabel("Sample #", color="#334155")
plt.tight_layout()
plt.savefig("outputs/signal_comparison.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("Saved outputs/signal_comparison.png")

# ── Plot 2: Recovery accuracy bar chart ──
from src.feature_engineering import compute_driving_features

results = {
    "clean":   compute_driving_features(clean),
    "noisy":   compute_driving_features(noisy),
    "cleaned": compute_driving_features(cleaned),
}

metrics = ["driving_style_score", "engine_stress_index", "avg_speed_kmh"]
x = np.arange(len(metrics))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
ax.set_facecolor("white")

clean_vals   = [results["clean"][m] for m in metrics]
noisy_vals   = [results["noisy"][m] for m in metrics]
cleaned_vals = [results["cleaned"][m] for m in metrics]

ax.bar(x - width, clean_vals, width, label="Clean (GT)", color="#22c55e")
ax.bar(x, noisy_vals, width, label="Noisy", color="#ef4444")
ax.bar(x + width, cleaned_vals, width, label="Cleaned", color="#3b82f6")

for i, (c, n_, cl) in enumerate(zip(clean_vals, noisy_vals, cleaned_vals)):
    ax.text(i - width, c + 0.5, f"{c:.2f}", ha="center", fontsize=8, color="#1e293b")
    ax.text(i,         n_ + 0.5, f"{n_:.2f}", ha="center", fontsize=8, color="#1e293b")
    ax.text(i + width, cl + 0.5, f"{cl:.2f}", ha="center", fontsize=8, color="#1e293b")

for i, m in enumerate(metrics):
    recovery = 100 * (1 - abs(clean_vals[i]-cleaned_vals[i]) / max(abs(clean_vals[i]), 1e-6))
    ax.annotate(f"{recovery:.1f}% recovered",
                 xy=(i+width, cleaned_vals[i]), xytext=(i+width, cleaned_vals[i]+max(clean_vals)*0.08),
                 ha="center", fontsize=9, color="#22c55e", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#22c55e"))

ax.set_xticks(x)
ax.set_xticklabels(metrics, color="#1e293b", fontsize=9)
ax.tick_params(colors="#475569")
for spine in ax.spines.values():
    spine.set_color("#cbd5e1")
ax.set_title("Feature Recovery: Clean vs Noisy vs Cleaned", color="#1e293b", fontsize=13)
ax.legend(facecolor="white", edgecolor="#cbd5e1", labelcolor="#1e293b")
ax.grid(axis="y", color="#e2e8f0", linewidth=0.8)
plt.tight_layout()
plt.savefig("outputs/recovery_metrics.png", dpi=150, bbox_inches="tight", facecolor="white")
print("Saved outputs/recovery_metrics.png")
plt.close()

# ── Plot 3: GPS Route with Geofencing ──
from src.geofence import GEOFENCES

gps_df = pd.read_csv("data/gps_geofenced.csv")

fig, ax = plt.subplots(figsize=(10, 10), facecolor="white")
ax.set_facecolor("white")

# Route line, colored by speed
sc = ax.scatter(gps_df["longitude"], gps_df["latitude"],
                 c=gps_df["speed_kmh"], cmap="viridis",
                 s=15, zorder=3)
ax.plot(gps_df["longitude"], gps_df["latitude"],
        color="#94a3b8", linewidth=0.8, alpha=0.4, zorder=2)

cbar = plt.colorbar(sc, ax=ax, shrink=0.7, label="Speed (km/h)")
cbar.ax.yaxis.label.set_color("#1e293b")
cbar.ax.tick_params(colors="#475569")

# Geofence circles
colors = {"Depot": "#22c55e", "City Zone": "#f59e0b", "Highway": "#3b82f6"}
for name, lat, lng, radius_km in GEOFENCES:
    circle = plt.Circle((lng, lat), radius_km / 111,
                         color=colors.get(name, "#ef4444"),
                         fill=True, alpha=0.12,
                         edgecolor=colors.get(name, "#ef4444"),
                         linewidth=2, zorder=1)
    ax.add_patch(circle)
    ax.text(lng, lat + radius_km/111 + 0.005, name,
            ha="center", fontsize=10, fontweight="bold",
            color=colors.get(name, "#ef4444"))

# Entry/exit markers
events = gps_df[gps_df["depot_event"].notna() & (gps_df["depot_event"] != "")]
for _, row in events.iterrows():
    marker_color = "#22c55e" if row["depot_event"] == "ENTRY" else "#ef4444"
    ax.scatter(row["longitude"], row["latitude"],
               s=200, marker="*", color=marker_color,
               edgecolor="black", linewidth=1, zorder=5,
               label=f"Depot {row['depot_event']}")

ax.set_xlabel("Longitude", color="#475569")
ax.set_ylabel("Latitude", color="#475569")
ax.set_title("Vehicle Route with Geofence Zones — Speed-Colored Track",
              color="#1e293b", fontsize=13)
ax.tick_params(colors="#475569")
for spine in ax.spines.values():
    spine.set_color("#cbd5e1")
ax.set_aspect("equal")

if not events.empty:
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(),
              facecolor="white", edgecolor="#cbd5e1", labelcolor="#1e293b")

plt.tight_layout()
plt.savefig("outputs/gps_route.png", dpi=150, bbox_inches="tight", facecolor="white")
print("Saved outputs/gps_route.png")
plt.close()

print("\nAll outputs generated successfully!")