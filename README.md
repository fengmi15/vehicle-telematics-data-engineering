# Vehicle Telematics Data Engineering Pipeline

End-to-end simulation and recovery pipeline for automotive CAN/OBD-II telemetry — covering signal generation, DBC-based encode/decode, realistic two-domain fault injection, a multi-stage cleaning pipeline, GPS/geofencing, and an interactive Streamlit dashboard.

## Overview

Real vehicle telemetry is noisy at two distinct layers: the **sensor layer** (faulty transducers, stuck ADCs, out-of-range readings) and the **bus layer** (dropped CAN frames, duplicate retransmissions, timestamp jitter). This project models both layers independently and builds a cleaning pipeline that recovers ground-truth driving features with **95–99% accuracy**.

## Architecture

```
can_simulator.py          → clean_drive.csv          (60-min drive cycle: city/highway/traffic/aggressive)
sensor_noise_injector.py  → sensor_noisy_drive.csv    (sensor glitches, stuck-at-value, out-of-range)
can_encoder.py             → raw_can_frames.csv        (DBC-encoded CAN frames)
bus_noise_injector.py      → raw_can_frames_noisy.csv  (dropped/duplicate frames, timestamp jitter)
can_decoder.py              → decoded_drive.csv / decoded_noisy_drive.csv  (DBC decode + 1Hz resample)
data_cleaner.py             → cleaned_drive.csv        (7-stage cleaning pipeline)
feature_engineering.py      → driving features & recovery metrics
gps_simulator.py → nmea_parser.py → geofence.py        → GPS route with depot entry/exit detection
generate_outputs.py         → outputs/*.png
dashboard.py                → Streamlit live dashboard
```

## Key Design Decisions

**Two physically distinct noise domains**
- *Sensor-level faults* (glitch spikes, stuck-at-value, out-of-range) are injected on physical values **before** CAN encoding — modeling a faulty transducer feeding bad data to the ECU.
- *Bus-level faults* (dropped frames, duplicates, timestamp jitter) are injected on raw CAN bytes **after** encoding, before decoding — modeling transmission-layer effects on the physical bus wire.
- The decoder is deterministic and simply reveals whatever survived transmission, faithfully reflecting both fault domains in the decoded dataset.

**DBC-based encode/decode**
- `dbc/vehicle.dbc` defines `ENGINE_DATA` (RPM, throttle, coolant temp, engine load) and `VEHICLE_SPEED` (speed, fuel level) with SAE J1939-style scaling/offsets.
- Full round-trip (physical → raw CAN bytes → physical) via `cantools`, proving understanding of the complete signal chain.

**7-stage cleaning pipeline**
1. Deduplication of retransmitted frames
2. Range validation (impossible values → NaN)
3. Stuck-at-value detection via rolling standard deviation
4. Outlier detection via z-score on rate-of-change
5. Linear interpolation for short gaps
6. Drop of unrecoverable rows
7. 1Hz resampling to align interleaved CAN messages

## Results — Feature Recovery Accuracy

| Metric | Ground Truth | Noisy | Cleaned | Recovery |
|---|---|---|---|---|
| Avg Speed (km/h) | 50.55 | 50.26 | 49.44 | 97.8% |
| Max Speed (km/h) | 102.18 | 102.18 | 102.18 | 100.0% |
| Avg RPM | 1885.4 | 1971.2 | 1927.6 | 97.8% |
| Idle Time % | 5.06 | 4.81 | 5.23 | 96.6% |
| Engine Stress Index | 0.3666 | 0.3707 | 0.3696 | 99.2% |

Harsh braking/acceleration counts remain sensitive to dropped-frame artifacts — a known limitation reflecting real transmission-layer loss, addressed in production systems via redundancy and error correction.

## GPS & Geofencing

- Simulated GPS route (Pune, India) encoded as real NMEA GPRMC sentences, parsed via `pynmea2`.
- Haversine-distance geofencing detects depot ENTRY/EXIT events across Depot, City Zone, and Highway zones.
- Speed-colored route visualization (`outputs/gps_route.png`).

## Dashboard

Dockerized Streamlit app (`dashboard.py`) showing:
- Side-by-side driving feature comparison (Clean / Noisy / Cleaned)
- Interactive signal comparison charts (Speed, RPM, Coolant Temp)
- Feature recovery accuracy table

## Tech Stack

Python, pandas, numpy, cantools, pynmea2, matplotlib, plotly, Streamlit, Docker

## Running the Project

```bash
# Setup
python -m venv venv
source venv/bin/activate   # or venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt

# Run pipeline in order
python src/can_simulator.py
python src/sensor_noise_injector.py
python src/can_encoder.py
python src/bus_noise_injector.py
python src/can_decoder.py
python src/data_cleaner.py
python src/feature_engineering.py
python src/gps_simulator.py
python src/nmea_parser.py
python src/geofence.py

# Generate plots
python generate_outputs.py

# Run dashboard
streamlit run dashboard.py

# Or via Docker
docker compose up --build
```

## Project Structure

```
vehicle-telematics-data-engineering/
├── dbc/vehicle.dbc
├── src/
│   ├── can_simulator.py
│   ├── sensor_noise_injector.py
│   ├── can_encoder.py
│   ├── bus_noise_injector.py
│   ├── can_decoder.py
│   ├── data_cleaner.py
│   ├── feature_engineering.py
│   ├── gps_simulator.py
│   ├── nmea_parser.py
│   └── geofence.py
├── outputs/
├── dashboard.py
├── generate_outputs.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```
