import numpy as np
import pandas as pd

np.random.seed(11)

def inject_bus_noise(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Injects bus-transmission-level faults — these occur AFTER the ECU 
    has correctly encoded the frame, but DURING transmission on the 
    physical bus wire.

    1. Dropped frames — bus congestion, arbitration loss, electrical noise
    2. Duplicate frames — retransmission after CAN error frame / ACK failure
    3. Timestamp jitter — bus arbitration delay, receiver buffering latency
    """
    noisy = raw_df.copy()
    n = len(noisy)

    # 1. Dropped frames (~3%) — bus congestion / arbitration loss
    drop_idx = np.random.choice(noisy.index, size=int(n * 0.03), replace=False)
    noisy = noisy.drop(drop_idx).reset_index(drop=True)
    n = len(noisy)

    # 2. Timestamp jitter (+-50ms) — arbitration delay / receiver buffering
    noisy["timestamp"] = noisy["timestamp"] + np.random.normal(0, 50, n).astype(np.int64)
    noisy = noisy.sort_values("timestamp").reset_index(drop=True)

    # 3. Duplicate frames (~1.5%) — retransmission after error frame
    dup_idx = np.random.choice(noisy.index, size=int(n * 0.015), replace=False)
    duplicates = noisy.loc[dup_idx].copy()
    noisy = pd.concat([noisy, duplicates]).sort_values("timestamp").reset_index(drop=True)

    return noisy


if __name__ == "__main__":
    raw_df = pd.read_csv("data/raw_can_frames.csv")
    noisy_raw = inject_bus_noise(raw_df)
    noisy_raw.to_csv("data/raw_can_frames_noisy.csv", index=False)
    print(f"Clean raw frames: {len(raw_df)}")
    print(f"Noisy raw frames: {len(noisy_raw)} (after drops + duplicates)")
    print("-> data/raw_can_frames_noisy.csv  (feed this into can_decoder.py)")