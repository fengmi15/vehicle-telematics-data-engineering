import cantools
import pandas as pd
import numpy as np

db = cantools.database.load_file("dbc/vehicle.dbc")

def encode_drive_to_raw_can(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts physical sensor values into raw CAN frames (arbitration ID + 
    data bytes) using the DBC definitions. This simulates what actually 
    appears on the CAN bus.
    """
    engine_msg = db.get_message_by_name("ENGINE_DATA")
    speed_msg  = db.get_message_by_name("VEHICLE_SPEED")

    rows = []
    for _, row in df.iterrows():
        # Encode ENGINE_DATA frame
        engine_data = engine_msg.encode({
            "EngineRPM":   np.clip(row["rpm"], 0, 8000),
            "ThrottlePos": np.clip(row["throttle_pct"], 0, 100),
            "CoolantTemp": np.clip(row["coolant_temp"], -40, 150),
            "EngineLoad":  np.clip(row["engine_load"], 0, 100),
        })
        rows.append({
            "timestamp": row["timestamp"],
            "can_id": hex(engine_msg.frame_id),
            "dlc": engine_msg.length,
            "data_hex": engine_data.hex(),
        })

        # Encode VEHICLE_SPEED frame
        speed_data = speed_msg.encode({
            "VehicleSpeed": np.clip(row["speed_kmh"], 0, 300),
            "FuelLevel":    np.clip(row["fuel_level"], 0, 100),
        })
        rows.append({
            "timestamp": row["timestamp"],
            "can_id": hex(speed_msg.frame_id),
            "dlc": speed_msg.length,
            "data_hex": speed_data.hex(),
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = pd.read_csv("data/sensor_noisy_drive.csv")
    raw_can = encode_drive_to_raw_can(df)
    raw_can.to_csv("data/raw_can_frames.csv", index=False)
    print(f"Encoded {len(raw_can)} raw CAN frames -> data/raw_can_frames.csv")
    print(raw_can.head(6))