import numpy as np
import pandas as pd

def compute_driving_features(df: pd.DataFrame) -> dict:
    """
    Computes telematics features used for driving style classification
    and fleet analytics.
    """
    df = df.dropna().reset_index(drop=True)
    
    # Interpolate any remaining gaps BEFORE computing derivatives
    df[["speed_kmh","rpm"]] = df[["speed_kmh","rpm"]].interpolate(method="linear", limit=5)
    df = df.dropna().reset_index(drop=True)

    speed = df["speed_kmh"].values
    rpm   = df["rpm"].values
    throttle = df["throttle_pct"].values

    # Acceleration (derivative of speed, m/s^2 approx via km/h per sample)
    accel = np.diff(speed)

    # Harsh braking: deceleration < -8 km/h per sample
    harsh_braking = int(np.sum(accel < -8))

    # Harsh acceleration: accel > 8 km/h per sample
    harsh_accel = int(np.sum(accel > 8))

    # Idle time: speed near 0 and RPM near idle
    idle_mask = (speed < 2) & (rpm < 1100)
    idle_pct = round(100 * idle_mask.sum() / len(df), 2)

    # Driving style score: weighted composite (0=eco, 1=aggressive)
    throttle_var = np.std(throttle)
    rpm_var = np.std(rpm)
    style_score = round(
        np.clip(
            (harsh_braking + harsh_accel) / len(df) * 50
            + throttle_var / 100
            + rpm_var / 3000,
            0, 1
        ), 4
    )

    # Engine stress index: composite of RPM, coolant, load
    stress_index = round(
        np.mean(
            (rpm / 8000) * 0.4 +
            (df["coolant_temp"].values / 130) * 0.3 +
            (df["engine_load"].values / 100) * 0.3
        ), 4
    )

    return {
        "avg_speed_kmh":    round(float(np.mean(speed)), 2),
        "max_speed_kmh":    round(float(np.max(speed)), 2),
        "avg_rpm":          round(float(np.mean(rpm)), 1),
        "harsh_braking_events":   harsh_braking,
        "harsh_acceleration_events": harsh_accel,
        "idle_time_pct":    idle_pct,
        "driving_style_score": style_score,
        "engine_stress_index": stress_index,
        "sample_count":     len(df),
    }


if __name__ == "__main__":
    import json
    clean   = pd.read_csv("data/clean_drive.csv")
    noisy   = pd.read_csv("data/decoded_noisy_drive.csv")
    cleaned = pd.read_csv("data/cleaned_drive.csv")

    results = {
        "clean":   compute_driving_features(clean),
        "noisy":   compute_driving_features(noisy),
        "cleaned": compute_driving_features(cleaned),
    }

    print(json.dumps(results, indent=2))

    # Recovery percentage (cleaned vs clean ground truth)
    print("\n--- Recovery Accuracy (cleaned vs clean ground truth) ---")
    for key in results["clean"]:
        if key == "sample_count":
            continue
        gt = results["clean"][key]
        cl = results["cleaned"][key]
        ns = results["noisy"][key]
        if gt == 0:
            continue
        recovery = round(100 * (1 - abs(gt - cl) / max(abs(gt), 1e-6)), 1)
        noisy_err = round(100 * (1 - abs(gt - ns) / max(abs(gt), 1e-6)), 1)
        print(f"{key:28s} | GT: {gt:>8} | Noisy: {ns:>8} ({noisy_err}%) | Cleaned: {cl:>8} ({recovery}% recovery)")