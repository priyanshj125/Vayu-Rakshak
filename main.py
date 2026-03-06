"""
main.py — FastAPI application for the Vayu-Rakshak Air Quality Monitoring System.

Endpoints:
  POST /register_sensor    — Register a new sensor; returns auto-generated api_key
  POST /ingest             — Ingest a sensor reading (requires x-api-key header)
  POST /predict            — Run PM2.5 correction inference via PyTorch model
  GET  /sensors            — List all registered sensors
  GET  /readings           — All readings joined with sensor location (for heatmap)
  GET  /readings/{sensor_id} — Readings for a specific sensor (for analytics)
  GET  /health             — Health check

Security:
  The /ingest endpoint validates x-api-key against the SensorRegistry table.

Background Tasks:
  On ingest, if is_anomaly==1 or pm2p5_corrected > 150, a background task logs
  a critical 🚨 alert to the server console and (optionally) sends notifications.
"""

import logging
import uuid
from datetime import datetime, date
from typing import List, Optional

from fastapi import FastAPI, Depends, HTTPException, Header, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from database import (
    SensorRegistry,
    SensorReadings,
    init_db,
    get_db,
)
from model_utils import predict as model_predict, load_model

# ─────────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Vayu-Rakshak Air Quality API",
    description=(
        "Real-time air quality monitoring system with sensor registration, "
        "data ingestion, anomaly alerting, and PM2.5 correction inference."
    ),
    version="1.0.0",
)

# Allow Streamlit (localhost:8501) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Startup / Shutdown
# ─────────────────────────────────────────────
@app.on_event("startup")
async def on_startup():
    init_db()
    # Pre-warm the model so the first /predict call is fast
    try:
        load_model()
        logger.info("🔥 PyTorch model pre-warmed and ready.")
    except FileNotFoundError as e:
        logger.warning(f"⚠️  Model not loaded at startup: {e}")


# ─────────────────────────────────────────────
# Pydantic Schemas
# ─────────────────────────────────────────────

class SensorRegisterRequest(BaseModel):
    sensor_id:     str = Field(..., example="ARI-1885")
    location_name: str = Field(..., example="Kashmiri Gate, Delhi")
    lat:           float = Field(..., example=28.6667)
    long:          float = Field(..., example=77.2283)
    installation_date: Optional[str] = Field(None, example="2025-01-15")

class SensorRegisterResponse(BaseModel):
    message:    str
    sensor_id:  str
    api_key:    str

class SensorInfo(BaseModel):
    sensor_id:         str
    location_name:     str
    lat:               float
    long:              float
    installation_date: Optional[str]

    class Config:
        from_attributes = True

class IngestRequest(BaseModel):
    sensor_id:        str    = Field(..., example="ARI-1885")
    timestamp:        str    = Field(..., example="2026-03-06 00:00:00")
    temperature:      float  = Field(..., example=25.4)
    humidity:         float  = Field(..., example=60.2)
    pm2p5_raw:        float  = Field(..., example=150.5)
    pm2p5_corrected:  float  = Field(..., example=142.1)
    is_anomaly:       int    = Field(0, example=0)
    is_failure:       int    = Field(0, example=0)

class IngestResponse(BaseModel):
    status:     str
    reading_id: int
    sensor_id:  str

class PredictRequest(BaseModel):
    features: List[float] = Field(
        ...,
        min_items=7,
        max_items=7,
        example=[150.0, 60.0, 25.0, 1012.0, 2.0, 30.0, 145.0],
        description="7 features: [pm2p5, humidity, temp, pressure, wind, cloud, valore_originale]",
    )

class PredictResponse(BaseModel):
    predicted_pm2p5_corrected: float
    unit: str = "µg/m³"

class ReadingOut(BaseModel):
    id:              int
    sensor_id:       str
    timestamp:       str
    temperature:     Optional[float]
    humidity:        Optional[float]
    pm2p5_raw:       Optional[float]
    pm2p5_corrected: Optional[float]
    is_anomaly:      int
    is_failure:      int
    lat:             Optional[float]
    long:            Optional[float]
    location_name:   Optional[str]

    class Config:
        from_attributes = True


# ─────────────────────────────────────────────
# Background Task: Alert
# ─────────────────────────────────────────────

def alert_high_pollution(sensor_id: str, pm2p5_corrected: float, is_anomaly: int):
    """
    Background task triggered when a reading is flagged as anomalous or
    has critically high PM2.5 levels. Logs a 🚨 alert to the console.
    In production this would trigger push notifications / PagerDuty / SMS.
    """
    reasons = []
    if is_anomaly == 1:
        reasons.append("ANOMALY DETECTED")
    if pm2p5_corrected > 150:
        reasons.append(f"HIGH PM2.5 ({pm2p5_corrected:.1f} µg/m³ > 150 threshold)")

    reason_str = " | ".join(reasons)
    logger.critical(
        f"🚨 ALERT: {reason_str} at Sensor [{sensor_id}]! "
        f"Immediate investigation required."
    )
    # ── Future hooks ──────────────────────────────────────────────────────
    # send_email_alert(sensor_id, reasons)
    # send_slack_message(sensor_id, reasons)
    # create_incident_ticket(sensor_id, reasons)


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

# ── Health ────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    return {"status": "ok", "service": "Vayu-Rakshak API"}


# ── Sensor Registration ───────────────────────────────────────────────────

@app.post(
    "/register_sensor",
    response_model=SensorRegisterResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Sensors"],
    summary="Register a new IoT sensor",
)
def register_sensor(payload: SensorRegisterRequest, db: Session = Depends(get_db)):
    """
    Register a new sensor in the network.
    Returns a unique `api_key` (UUID) that must be sent as the `x-api-key`
    header with every subsequent /ingest call from this sensor.
    """
    existing = db.query(SensorRegistry).filter(
        SensorRegistry.sensor_id == payload.sensor_id
    ).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Sensor '{payload.sensor_id}' is already registered. "
                   f"Use the existing api_key for ingestion.",
        )

    inst_date = date.today()
    if payload.installation_date:
        try:
            inst_date = datetime.strptime(payload.installation_date, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="installation_date must be in YYYY-MM-DD format.",
            )

    new_key = str(uuid.uuid4())
    sensor = SensorRegistry(
        sensor_id=payload.sensor_id,
        location_name=payload.location_name,
        lat=payload.lat,
        long=payload.long,
        installation_date=inst_date,
        api_key=new_key,
    )
    db.add(sensor)
    db.commit()
    db.refresh(sensor)

    logger.info(f"✅ Sensor registered: {sensor.sensor_id} @ {sensor.location_name}")
    return SensorRegisterResponse(
        message="Sensor registered successfully. Store the api_key securely.",
        sensor_id=sensor.sensor_id,
        api_key=new_key,
    )


# ── List Sensors ───────────────────────────────────────────────────────────

@app.get(
    "/sensors",
    response_model=List[SensorInfo],
    tags=["Sensors"],
    summary="List all registered sensors",
)
def list_sensors(db: Session = Depends(get_db)):
    sensors = db.query(SensorRegistry).all()
    return [
        SensorInfo(
            sensor_id=s.sensor_id,
            location_name=s.location_name,
            lat=s.lat,
            long=s.long,
            installation_date=str(s.installation_date) if s.installation_date else None,
        )
        for s in sensors
    ]


# ── Data Ingestion ─────────────────────────────────────────────────────────

@app.post(
    "/ingest",
    response_model=IngestResponse,
    tags=["Data Ingestion"],
    summary="Ingest a sensor reading (requires x-api-key header)",
)
def ingest_reading(
    payload: IngestRequest,
    background_tasks: BackgroundTasks,
    x_api_key: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    """
    Accept an air quality reading from a registered sensor.

    **Authentication**: The request must include an `x-api-key` header whose
    value matches the `api_key` recorded for the given `sensor_id`.

    **Background alerting**: If the reading is anomalous (`is_anomaly=1`) or if
    `pm2p5_corrected > 150 µg/m³`, a background task logs a 🚨 alert.
    """
    # ── Validate API key ──────────────────────────────────────────────────
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing x-api-key header. Authenticate with your sensor's API key.",
        )

    sensor = db.query(SensorRegistry).filter(
        SensorRegistry.sensor_id == payload.sensor_id
    ).first()

    if not sensor:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sensor '{payload.sensor_id}' not found. Register it first.",
        )

    if sensor.api_key != x_api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid api_key for this sensor.",
        )

    # ── Parse timestamp ───────────────────────────────────────────────────
    try:
        ts = datetime.strptime(payload.timestamp, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="timestamp must be in 'YYYY-MM-DD HH:MM:SS' format.",
        )

    # ── Persist reading ───────────────────────────────────────────────────
    reading = SensorReadings(
        sensor_id=payload.sensor_id,
        timestamp=ts,
        temperature=payload.temperature,
        humidity=payload.humidity,
        pm2p5_raw=payload.pm2p5_raw,
        pm2p5_corrected=payload.pm2p5_corrected,
        is_anomaly=payload.is_anomaly,
        is_failure=payload.is_failure,
    )
    db.add(reading)
    db.commit()
    db.refresh(reading)

    # ── Background alert if critical ──────────────────────────────────────
    should_alert = (payload.is_anomaly == 1) or (payload.pm2p5_corrected > 150)
    if should_alert:
        background_tasks.add_task(
            alert_high_pollution,
            payload.sensor_id,
            payload.pm2p5_corrected,
            payload.is_anomaly,
        )

    logger.info(
        f"📥 Reading ingested: sensor={payload.sensor_id} "
        f"pm2p5_corrected={payload.pm2p5_corrected} "
        f"anomaly={payload.is_anomaly}"
    )
    return IngestResponse(
        status="success",
        reading_id=reading.id,
        sensor_id=reading.sensor_id,
    )


# ── All Readings (for heatmap & surveillance) ──────────────────────────────

@app.get(
    "/readings",
    response_model=List[ReadingOut],
    tags=["Readings"],
    summary="Fetch all readings joined with sensor coordinates",
)
def get_all_readings(
    anomaly_only: bool = False,
    limit: int = 5000,
    db: Session = Depends(get_db),
):
    """
    Returns sensor readings joined with their geographic coordinates.
    Use `anomaly_only=true` to return only anomalous / failed readings.
    """
    query = (
        db.query(SensorReadings, SensorRegistry)
        .join(SensorRegistry, SensorReadings.sensor_id == SensorRegistry.sensor_id)
    )
    if anomaly_only:
        query = query.filter(
            (SensorReadings.is_anomaly == 1) | (SensorReadings.is_failure == 1)
        )

    rows = query.order_by(SensorReadings.timestamp.desc()).limit(limit).all()

    return [
        ReadingOut(
            id=r.id,
            sensor_id=r.sensor_id,
            timestamp=str(r.timestamp),
            temperature=r.temperature,
            humidity=r.humidity,
            pm2p5_raw=r.pm2p5_raw,
            pm2p5_corrected=r.pm2p5_corrected,
            is_anomaly=r.is_anomaly,
            is_failure=r.is_failure,
            lat=s.lat,
            long=s.long,
            location_name=s.location_name,
        )
        for r, s in rows
    ]


# ── Readings by Sensor (for analytics tab) ────────────────────────────────

@app.get(
    "/readings/{sensor_id}",
    response_model=List[ReadingOut],
    tags=["Readings"],
    summary="Fetch readings for a specific sensor",
)
def get_sensor_readings(
    sensor_id: str,
    limit: int = 2000,
    db: Session = Depends(get_db),
):
    sensor = db.query(SensorRegistry).filter(
        SensorRegistry.sensor_id == sensor_id
    ).first()
    if not sensor:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sensor '{sensor_id}' not found.",
        )

    rows = (
        db.query(SensorReadings)
        .filter(SensorReadings.sensor_id == sensor_id)
        .order_by(SensorReadings.timestamp.asc())
        .limit(limit)
        .all()
    )

    return [
        ReadingOut(
            id=r.id,
            sensor_id=r.sensor_id,
            timestamp=str(r.timestamp),
            temperature=r.temperature,
            humidity=r.humidity,
            pm2p5_raw=r.pm2p5_raw,
            pm2p5_corrected=r.pm2p5_corrected,
            is_anomaly=r.is_anomaly,
            is_failure=r.is_failure,
            lat=sensor.lat,
            long=sensor.long,
            location_name=sensor.location_name,
        )
        for r in rows
    ]


# ── Predict ────────────────────────────────────────────────────────────────

@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["ML Inference"],
    summary="Predict corrected PM2.5 using the trained PyTorch model",
)
def predict_endpoint(payload: PredictRequest):
    """
    Accepts exactly 7 features in order:
    `[pm2p5, humidity, temp, pressure, wind, cloud, valore_originale]`

    Returns the model's corrected PM2.5 prediction.
    """
    try:
        result = model_predict(payload.features)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model not available: {e}",
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {e}",
        )

    return PredictResponse(predicted_pm2p5_corrected=result)
