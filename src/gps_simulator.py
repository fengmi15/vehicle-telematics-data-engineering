import numpy as np
import pandas as pd

def simulate_gps_route(n_samples: int, freq_hz=1):
    """
    Simulates a GPS route matching the 4-phase drive cycle.
    Uses real Pune-area coordinates.
    """
    # Waypoints: (lat, lng) - Pune area
    waypoints = [
        (18.5204, 73.8567),  # City start
        (18.5500, 73.8200),  # Highway entry
        (18.6200, 73.7800),  # Highway midpoint
        (18.6500, 73.7500),  # Highway exit
        (18.6450, 73.7600),  # Depot / destination
    ]

    n_segments = len(waypoints) - 1
    samples_per_segment = n_samples // n_segments

    lats, lngs = [], []
    for i in range(n_segments):
        lat1, lng1 = waypoints[i]
        lat2, lng2 = waypoints[i+1]
        for s in range(samples_per_segment):
            t = s / samples_per_segment
            # Add small GPS noise (realistic receiver jitter ~5-10m)
            noise = np.random.normal(0, 0.00005, 2)
            lats.append(lat1 + (lat2-lat1)*t + noise[0])
            lngs.append(lng1 + (lng2-lng1)*t + noise[1])

    # Pad to exact n_samples
    while len(lats) < n_samples:
        lats.append(lats[-1])
        lngs.append(lngs[-1])

    timestamps = np.arange(n_samples) * (1000 / freq_hz)

    return pd.DataFrame({
        "timestamp": timestamps.astype(np.int64),
        "latitude": np.round(lats[:n_samples], 6),
        "longitude": np.round(lngs[:n_samples], 6),
    })


if __name__ == "__main__":
    drive = pd.read_csv("data/clean_drive.csv")
    gps = simulate_gps_route(len(drive))
    gps["speed_kmh"] = drive["speed_kmh"].values

    gps.to_csv("data/gps_track.csv", index=False)
    print(f"Generated {len(gps)} GPS points -> data/gps_track.csv")
    print(gps.head(3))