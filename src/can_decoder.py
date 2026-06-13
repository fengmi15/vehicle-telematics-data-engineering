import cantools
import pandas as pd

db = cantools.database.load_file("dbc/vehicle.dbc")

def decode_raw_can(raw_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in raw_df.iterrows():
        ts = row["timestamp"]
        can_id = int(row["can_id"], 16)
        data = bytes.fromhex(row["data_hex"])

        try:
            msg = db.get_message_by_frame_id(can_id)
            decoded = msg.decode(data)
        except Exception:
            continue

        decoded["timestamp"] = ts
        rows.append(decoded)

    df = pd.DataFrame(rows)
    df = df.rename(columns={
        "EngineRPM": "rpm",
        "ThrottlePos": "throttle_pct",
        "CoolantTemp": "coolant_temp",
        "EngineLoad": "engine_load",
        "VehicleSpeed": "speed_kmh",
        "FuelLevel": "fuel_level",
    })
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Fill gaps left by the other message type's columns
    df = df.ffill().bfill()

    # Resample to 1Hz — collapse the two interleaved messages per timestamp
    # back into one row per second (take last value within each second)
    df["t_sec"] = (df["timestamp"] // 1000)
    df = df.groupby("t_sec").last().reset_index()

    # Reindex to continuous seconds so dropped-frame gaps become NaN
    # (data_cleaner.py will interpolate/handle these)
    full_range = pd.DataFrame({"t_sec": range(int(df["t_sec"].min()), int(df["t_sec"].max()) + 1)})
    df = full_range.merge(df, on="t_sec", how="left")
    
    df["timestamp"] = df["t_sec"] * 1000
    df = df.drop(columns=["t_sec"])

    return df
if __name__ == "__main__":
    # Decode clean raw frames
    raw_df = pd.read_csv("data/raw_can_frames.csv")
    decoded = decode_raw_can(raw_df)
    decoded.to_csv("data/decoded_drive.csv", index=False)
    print(f"Decoded {len(decoded)} samples from clean raw frames -> data/decoded_drive.csv")
    
    # Decode noisy raw frames
    raw_noisy_df = pd.read_csv("data/raw_can_frames_noisy.csv")
    decoded_noisy = decode_raw_can(raw_noisy_df)
    decoded_noisy.to_csv("data/decoded_noisy_drive.csv", index=False)
    print(f"Decoded {len(decoded_noisy)} samples from noisy raw frames -> data/decoded_noisy_drive.csv")