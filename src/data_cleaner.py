import numpy as np
import pandas as pd

def clean_can_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    7-stage cleaning pipeline addressing each injected error class:
    1. Deduplication
    2. Range validation
    3. Stuck-at-value detection
    4. Outlier detection (z-score on rate of change)
    5. Interpolation for gaps
    6. Drop unrecoverable NaN rows
    7. Keep timestamps as-is (already realistic from bus_noise_injector)
    """
    cleaned = df.copy()

    # 1. Remove exact duplicate frames
    cleaned = cleaned.drop_duplicates(subset=["timestamp", "rpm", "speed_kmh"])
    cleaned = cleaned.sort_values("timestamp").reset_index(drop=True)

    # 2. Range validation -> NaN for impossible values
    cleaned.loc[cleaned["speed_kmh"] < 0, "speed_kmh"] = np.nan
    cleaned.loc[cleaned["rpm"] > 8000, "rpm"] = np.nan
    cleaned.loc[cleaned["coolant_temp"] > 130, "coolant_temp"] = np.nan

    # 3. Stuck-at-value detection (rolling std == 0 over window of 10)
    rolling_std = cleaned["coolant_temp"].rolling(10, min_periods=5).std()
    cleaned.loc[rolling_std == 0, "coolant_temp"] = np.nan

    # 4. Outlier detection via z-score on rate of change (RPM)
    rpm_diff = cleaned["rpm"].diff()
    z = (rpm_diff - rpm_diff.mean()) / rpm_diff.std()
    cleaned.loc[z.abs() > 3, "rpm"] = np.nan

    # 5. Interpolate short gaps (max 3 consecutive NaNs)
    cleaned[["speed_kmh","rpm","coolant_temp","throttle_pct","engine_load"]] = (
        cleaned[["speed_kmh","rpm","coolant_temp","throttle_pct","engine_load"]]
        .interpolate(method="linear", limit=3)
    )

    # 6. Drop remaining unrecoverable rows
    cleaned = cleaned.dropna().reset_index(drop=True)

    return cleaned


if __name__ == "__main__":
    noisy_df = pd.read_csv("data/decoded_noisy_drive.csv")
    cleaned_df = clean_can_data(noisy_df)
    cleaned_df.to_csv("data/cleaned_drive.csv", index=False)
    print(f"Noisy samples:  {len(noisy_df)}")
    print(f"Cleaned samples: {len(cleaned_df)}")
    print(f"Saved -> data/cleaned_drive.csv")