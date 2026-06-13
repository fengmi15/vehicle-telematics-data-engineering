import pynmea2
import pandas as pd

def parse_nmea_log(gps_df: pd.DataFrame) -> pd.DataFrame:
    """
    Parses NMEA GPRMC sentences into structured lat/lng/speed.
    """
    parsed = []
    for _, row in gps_df.iterrows():
        # Create NMEA sentence from coordinates
        lat_deg = int(abs(row["latitude"]))
        lat_min = (abs(row["latitude"]) - lat_deg) * 60
        lat_str = f"{lat_deg:02d}{lat_min:07.4f}"
        lat_dir = "N" if row["latitude"] >= 0 else "S"

        lng_deg = int(abs(row["longitude"]))
        lng_min = (abs(row["longitude"]) - lng_deg) * 60
        lng_str = f"{lng_deg:03d}{lng_min:07.4f}"
        lng_dir = "E" if row["longitude"] >= 0 else "W"

        speed_knots = row["speed_kmh"] / 1.852
        sentence = (
            f"$GPRMC,123519,A,{lat_str},{lat_dir},{lng_str},{lng_dir},"
            f"{speed_knots:.1f},054.7,191194,003.1,W"
        )
        checksum = 0
        for ch in sentence[1:]:
            checksum ^= ord(ch)
        sentence = f"{sentence}*{checksum:02X}"

        try:
            msg = pynmea2.parse(sentence)
            parsed.append({
                "timestamp": row["timestamp"],
                "latitude": float(msg.latitude),
                "longitude": float(msg.longitude),
                "speed_kmh": float(msg.spd_over_grnd) * 1.852,
                "status": msg.status,
            })
        except Exception:
            continue

    return pd.DataFrame(parsed)


if __name__ == "__main__":
    gps_df = pd.read_csv("data/gps_track.csv")
    parsed = parse_nmea_log(gps_df)
    parsed.to_csv("data/gps_parsed.csv", index=False)
    print(f"Parsed {len(parsed)} NMEA sentences -> data/gps_parsed.csv")
    print(parsed.head())