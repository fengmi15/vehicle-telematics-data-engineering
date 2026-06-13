import numpy as np
import pandas as pd

np.random.seed(7)

def inject_sensor_noise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Injects sensor/transducer-level faults — these occur BEFORE the ECU 
    encodes values into CAN frames, because the sensor itself is producing 
    bad readings.

    1. Sensor glitch (transient spike) — loose connector, EMI on signal wire
    2. Stuck-at-value — frozen ADC reading or corroded connector
    3. Out-of-range value — failed sensor producing physically impossible reading
    """
    noisy = df.copy()
    n = len(noisy)

    # 1. Sensor glitch on RPM (~2%) — transient spike 2-5x
    glitch_idx = np.random.choice(noisy.index, size=int(n * 0.02), replace=False)
    noisy.loc[glitch_idx, "rpm"] = (
        noisy.loc[glitch_idx, "rpm"] * np.random.uniform(2, 5, len(glitch_idx))
    ).clip(upper=8000).astype(int)

    # 2. Stuck-at-value on coolant_temp — frozen ADC for 15 samples
    stuck_start = np.random.randint(0, n - 20)
    frozen_val = noisy.iloc[stuck_start]["coolant_temp"]
    noisy.loc[stuck_start:stuck_start + 15, "coolant_temp"] = frozen_val

    # 3. Out-of-range speed sensor fault (~1%) — wheel speed sensor failure
    bad_idx = np.random.choice(noisy.index, size=int(n * 0.01), replace=False)
    noisy.loc[bad_idx, "speed_kmh"] = -1.0

    return noisy


if __name__ == "__main__":
    clean_df = pd.read_csv("data/clean_drive.csv")
    sensor_noisy_df = inject_sensor_noise(clean_df)
    sensor_noisy_df.to_csv("data/sensor_noisy_drive.csv", index=False)
    print(f"Applied sensor-level faults to {len(sensor_noisy_df)} samples")
    print("-> data/sensor_noisy_drive.csv  (feed this into can_encoder.py)")