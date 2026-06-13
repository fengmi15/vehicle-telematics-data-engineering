import numpy as np
import pandas as pd

# Define geofence zones: (name, center_lat, center_lng, radius_km)
GEOFENCES = [
    ("Depot",      18.6450, 73.7600, 0.5),
    ("City Zone",  18.5204, 73.8567, 2.0),
    ("Highway",    18.6000, 73.8000, 5.0),
]

def haversine_km(lat1, lng1, lat2, lng2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlng = np.radians(lng2 - lng1)
    a = (np.sin(dlat/2)**2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
         np.sin(dlng/2)**2)
    return R * 2 * np.arcsin(np.sqrt(a))


def check_geofences(gps_df: pd.DataFrame) -> pd.DataFrame:
    """
    Checks each GPS point against defined geofence zones.
    Detects ENTRY and EXIT events.
    """
    df = gps_df.copy()

    for name, clat, clng, radius in GEOFENCES:
        dist = haversine_km(df["latitude"].values, df["longitude"].values, clat, clng)
        df[f"in_{name.replace(' ','_')}"] = dist <= radius

    # Detect entry/exit events for the Depot zone
    df["depot_state"] = df["in_Depot"].map({True: "INSIDE", False: "OUTSIDE"})
    df["depot_event"] = ""
    transitions = df["in_Depot"].astype(int).diff()
    df.loc[transitions == 1, "depot_event"] = "ENTRY"
    df.loc[transitions == -1, "depot_event"] = "EXIT"

    return df


if __name__ == "__main__":
    gps_df = pd.read_csv("data/gps_parsed.csv")
    result = check_geofences(gps_df)
    result.to_csv("data/gps_geofenced.csv", index=False)

    events = result[result["depot_event"] != ""]
    print(f"Geofence events detected: {len(events)}")
    print(events[["timestamp","latitude","longitude","depot_event"]])