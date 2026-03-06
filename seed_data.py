"""
seed_data.py — Seed the Vayu-Rakshak database with a massive sensor network.

This script procedurally generates 100+ simulated sensors around Roorkee,
Uttarakhand, using data from the existing ari-1727.csv file as a base template,
with slight random noise added for realistic variance between sensors.

Each sensor is spaced around the core Roorkee coordinates:
Lat: 29.86, Long: 77.89

Run ONCE after starting the FastAPI server:
    python seed_data.py
"""

import os
import json
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime

API_BASE = "http://localhost:8000"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
KEYS_FILE = os.path.join(os.path.dirname(__file__), "api_keys.json")

# Base coordinate for Roorkee
ROORKEE_LAT = 29.865
ROORKEE_LON = 77.890

# Generate 100 sensors around Roorkee (roughly within a 15km radius)
# 1 degree lat/lon is ~111km. So 0.1 degree is ~11km.
NUM_SENSORS = 100
np.random.seed(42)  # For reproducible sensor locations

SENSORS = []
for i in range(1, NUM_SENSORS + 1):
    lat_offset = np.random.uniform(-0.15, 0.15)
    lon_offset = np.random.uniform(-0.15, 0.15)
    SENSORS.append({
        "sensor_id": f"VRK-{1000 + i}",
        "location_name": f"Roorkee Zone {i}",
        "lat": ROORKEE_LAT + lat_offset,
        "long": ROORKEE_LON + lon_offset,
        "csv": "ari-1727.csv",  # Use this CSV as the base template
    })

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def z_score_anomaly(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Return boolean mask: True where |z-score| > threshold."""
    mu, sigma = series.mean(), series.std()
    if sigma == 0:
        return pd.Series([False] * len(series))
    return ((series - mu) / sigma).abs() > threshold

def register_all_sensors() -> dict:
    api_keys = {}
    if os.path.isfile(KEYS_FILE):
        with open(KEYS_FILE) as f:
            api_keys = json.load(f)
        print(f"📂 Loaded {len(api_keys)} existing API keys")

    registered_count = 0
    print(f"📡 Registering {len(SENSORS)} sensors - this might take a moment...")
    
    # We use a session for keep-alive to make this much faster
    with requests.Session() as session:
        for s in SENSORS:
            sid = s["sensor_id"]
            if sid in api_keys:
                continue
            try:
                payload = {
                    "sensor_id":    sid,
                    "location_name":s["location_name"],
                    "lat":          s["lat"],
                    "long":         s["long"],
                }
                r = session.post(f"{API_BASE}/register_sensor", json=payload, timeout=10)
                if r.status_code == 201:
                    api_keys[sid] = r.json()["api_key"]
                    registered_count += 1
                elif r.status_code == 409:
                    pass
                else:
                    print(f"  ❌ {sid}: {r.status_code} — {r.text[:80]}")
            except Exception as e:
                print(f"❌ Cannot connect to {API_BASE}. Is FastAPI running? {e}")
                raise SystemExit(1)
                
    if registered_count > 0:
        print(f"  ✅ Registered {registered_count} new sensors.")
        with open(KEYS_FILE, "w") as f:
            json.dump(api_keys, f, indent=2)
        print(f"💾 API keys saved to {KEYS_FILE}\n")
    else:
        print(f"  ⏭️ All {len(SENSORS)} sensors already registered.\n")
        
    return api_keys

def ingest_all_sensors(api_keys: dict, max_rows: int = 24):
    """
    Ingest data for all sensors simultaneously.
    Instead of full CSVs (which would be 100 * 5000 = 500k rows = hours of runtime),
    we'll ingest the latest 'max_rows' readings per sensor so the map looks great
    immediately, with realistic temporal distribution.
    """
    csv_path = os.path.join(DATA_DIR, "ari-1727.csv")
    if not os.path.isfile(csv_path):
        print(f"  ⚠️  Base CSV not found: {csv_path} — cannot seed.")
        return

    # Load base template dataset
    print(f"📥 Loading base dataset from {csv_path}...")
    base_df = pd.read_csv(csv_path)
    
    # Just take a chunk of data (e.g., last 24 readings) to seed the dashboard quickly
    base_df = base_df.tail(max_rows).copy()
    base_df["is_anomaly"] = z_score_anomaly(base_df["pm2p5"], threshold=3.0).astype(int)
    base_df.loc[base_df["pm2p5"] > 150, "is_anomaly"] = 1
    base_df["is_failure"] = 0

    print(f"🔥 Generating and ingesting {max_rows} readings for {len(SENSORS)} sensors = {max_rows * len(SENSORS)} total points...")
    
    success, failed = 0, 0
    
    with requests.Session() as session:
        # Pre-calculate base timestamps
        timestamps = []
        for raw_ts in base_df["valid_at"]:
            try:
                # The data is from 2022, let's shift it to recent days to make it look alive
                dt_obj = datetime.strptime(str(raw_ts).strip(), "%Y-%m-%d %H:%M:%S")
                # Shift by ~4 years
                dt_obj = dt_obj.replace(year=2026, month=3)
                timestamps.append(dt_obj.strftime("%Y-%m-%d %H:%M:%S"))
            except ValueError:
                timestamps.append("2026-03-06 00:00:00")
                
        base_df["parsed_time"] = timestamps

        for idx, sensor in enumerate(SENSORS):
            sensor_id = sensor["sensor_id"]
            if sensor_id not in api_keys:
                continue
                
            headers = {"Content-Type": "application/json", "x-api-key": api_keys[sensor_id]}
            
            # Predict only once per base row to save massive time
            # Then add variance to the prediction per sensor
            
            for i, row in base_df.iterrows():
                try:
                    # Add random spatial noise to create unique data per sensor
                    noise_factor = np.random.normal(1.0, 0.15) # +/- 15% variance across city
                    
                    pm25_raw = float(row["pm2p5"]) * noise_factor if pd.notna(row["pm2p5"]) else 50.0
                    humidity = float(row["relative_humidity"]) if pd.notna(row["relative_humidity"]) else 60.0
                    temp     = float(row["temperature"]) if pd.notna(row["temperature"]) else 25.0
                    pressure = float(row["pressure"]) if pd.notna(row["pressure"]) else 1013.0
                    wind     = float(row["wind_speed"]) if pd.notna(row["wind_speed"]) else 2.0
                    cloud    = float(row["cloud_coverage"]) if pd.notna(row["cloud_coverage"]) else 30.0

                    # Mock /predict locally for speed or call real endpoint
                    pred_r = session.post(
                        f"{API_BASE}/predict",
                        json={"features": [pm25_raw, humidity, temp, pressure, wind, cloud, pm25_raw]},
                        timeout=5,
                    )
                    pm25_corrected = (
                        pred_r.json().get("predicted_pm2p5_corrected", pm25_raw)
                        if pred_r.status_code == 200 else pm25_raw
                    )

                    is_anomaly = 1 if pm25_raw > 150 else int(row["is_anomaly"])

                    payload = {
                        "sensor_id":       sensor_id,
                        "timestamp":       row["parsed_time"],
                        "temperature":     round(temp, 2),
                        "humidity":        round(humidity, 2),
                        "pm2p5_raw":       round(pm25_raw, 4),
                        "pm2p5_corrected": round(float(pm25_corrected), 4),
                        "is_anomaly":      is_anomaly,
                        "is_failure":      int(row["is_failure"]),
                    }

                    r = session.post(f"{API_BASE}/ingest", json=payload, headers=headers, timeout=5)
                    if r.status_code == 200:
                        success += 1
                    else:
                        failed += 1
                except Exception as e:
                    failed += 1
                    
            if (idx + 1) % 10 == 0:
                print(f"    … Finished sensor {idx+1}/{len(SENSORS)} ({success} total ingestions)")

    print(f"  ✅ Done: {success} ingested, {failed} failed.")

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("  Vayu-Rakshak — Massive Roorkee Dataset Seeder")
    print("  Generating 100 sensors across Roorkee, Uttarakhand")
    print("=" * 64)

    api_keys = register_all_sensors()
    ingest_all_sensors(api_keys, max_rows=24) # 24 hours of data * 100 sensors = 2400 points

    print("\n🎉 Seeding complete!")
    print("\n   Open http://localhost:8501 to explore the Streamlit dashboard.")

if __name__ == "__main__":
    main()
