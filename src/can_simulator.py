import numpy as np
import pandas as pd

np.random.seed(42)

def simulate_drive(duration_mins=60, freq_hz=1):
    """
    Simulates OBD-II PIDs over a realistic drive cycle.
    Phases: city (low speed, frequent stops) -> highway (high speed, steady) 
            -> traffic (idle/stop-go) -> aggressive (harsh accel/brake)
    """
    n_samples = duration_mins * 60 * freq_hz
    timestamps = np.arange(n_samples) * (1000 / freq_hz)  # ms

    speed   = np.zeros(n_samples)
    rpm     = np.zeros(n_samples)
    throttle = np.zeros(n_samples)
    coolant = np.zeros(n_samples)
    engine_load = np.zeros(n_samples)
    fuel_level = np.linspace(70, 65, n_samples)  # gradual fuel consumption

    # Phase boundaries (as fraction of total)
    phases = [
        ("city",       0.0, 0.30),
        ("highway",    0.30, 0.65),
        ("traffic",    0.65, 0.85),
        ("aggressive", 0.85, 1.0),
    ]

    coolant_val = 25.0  # cold start

    for phase_name, start_frac, end_frac in phases:
        start_idx = int(start_frac * n_samples)
        end_idx   = int(end_frac * n_samples)
        length = end_idx - start_idx

        for i in range(start_idx, end_idx):
            t = (i - start_idx) / max(length, 1)

            if phase_name == "city":
                target_speed = 30 + 20 * np.sin(t * np.pi * 6)  # stop-go pattern
                target_speed = max(0, target_speed)
                target_rpm   = 1200 + target_speed * 25
                target_throttle = 15 + target_speed * 0.5

            elif phase_name == "highway":
                target_speed = 90 + np.random.normal(0, 3)
                target_rpm   = 2200 + np.random.normal(0, 100)
                target_throttle = 35 + np.random.normal(0, 5)

            elif phase_name == "traffic":
                target_speed = 5 * (1 + np.sin(t * np.pi * 10))  # crawl
                target_rpm   = 900 + target_speed * 30
                target_throttle = 8 + target_speed * 1.5

            elif phase_name == "aggressive":
                # Sharp accel/decel cycles
                cycle = np.sin(t * np.pi * 4)
                target_speed = 60 + cycle * 40
                target_speed = max(0, target_speed)
                target_rpm   = 2000 + cycle * 2000
                target_throttle = 50 + cycle * 40

            # Smooth transitions (lag filter)
            prev_speed = speed[i-1] if i > 0 else 0
            prev_rpm   = rpm[i-1] if i > 0 else target_rpm
            prev_throttle = throttle[i-1] if i > 0 else target_throttle

            speed[i]    = prev_speed + (target_speed - prev_speed) * 0.3 + np.random.normal(0, 1)
            rpm[i]      = prev_rpm + (target_rpm - prev_rpm) * 0.3 + np.random.normal(0, 30)
            throttle[i] = np.clip(prev_throttle + (target_throttle - prev_throttle) * 0.3 + np.random.normal(0, 2), 0, 100)

            speed[i] = max(0, speed[i])
            rpm[i]   = max(700, rpm[i])  # idle RPM floor

            # Coolant warms up over time, stabilizes ~90C
            coolant_val += (90 - coolant_val) * 0.002 + np.random.normal(0, 0.1)
            coolant[i] = coolant_val

            # Engine load correlates with throttle + RPM
            engine_load[i] = np.clip(throttle[i] * 0.6 + (rpm[i]/8000)*40 + np.random.normal(0,2), 0, 100)

    df = pd.DataFrame({
        "timestamp":    timestamps.astype(np.int64),
        "speed_kmh":    np.round(speed, 2),
        "rpm":          np.round(rpm, 0).astype(int),
        "throttle_pct": np.round(throttle, 1),
        "coolant_temp": np.round(coolant, 1),
        "engine_load":  np.round(engine_load, 1),
        "fuel_level":   np.round(fuel_level, 2),
    })
    return df


if __name__ == "__main__":
    df = simulate_drive(duration_mins=60)
    df.to_csv("data/clean_drive.csv", index=False)
    print(f"Generated {len(df)} samples -> data/clean_drive.csv")
    print(df.describe())