import sqlite3  # Persist predictions for stats and drift checks.
import time  # Measure end-to-end inference latency.
from contextlib import asynccontextmanager  # Own startup/shutdown around the app.
from datetime import datetime, timezone  # Stamp predictions in UTC.

import mlflow  # Resolve the champion alias from the registry.
import mlflow.sklearn  # Load the serialized sklearn models.
import pandas as pd  # Build the single-row feature frame.
from fastapi import FastAPI, HTTPException, Response  # HTTP app, errors, and response headers.
from mlflow.tracking import MlflowClient  # Look up the loaded version numbers.
from pydantic import BaseModel, Field  # Validate telemetry and shape responses.

from src.config import (  # Shared thresholds, paths, and names.
    CASCADE_HIGH, CASCADE_LOW, DB_PATH, FAST_MODEL_NAME, FEATURE_COLS,
    HEAVY_MODEL_NAME, MLFLOW_TRACKING_URI, MODEL_ALIAS,
)


class TelemetryRequest(BaseModel):  # One telemetry sample with hard range checks.
    cpu_utilization: float = Field(..., ge=0.0, le=100.0)  # Percent CPU.
    memory_usage: float = Field(..., ge=0.0, le=100.0)  # Percent memory.
    disk_io: float = Field(..., ge=0.0)  # Disk I/O rate.
    network_latency: float = Field(..., ge=0.0)  # Network latency.
    process_count: int = Field(..., ge=1)  # At least one process.
    thread_count: int = Field(..., ge=1)  # At least one thread.
    context_switches: int = Field(..., ge=0)  # Non-negative counter.
    cache_miss_rate: float = Field(..., ge=0.0, le=1.0)  # Fraction, not percent.
    temperature: float = Field(..., ge=0.0, le=150.0)  # Degrees Celsius.
    power_consumption: float = Field(..., ge=0.0)  # Watts.
    uptime: float = Field(..., ge=0.0)  # Seconds.


class PredictionResponse(BaseModel):  # Stable contract for /predict consumers.
    prediction: int  # 0 = normal, 1 = anomaly.
    status_label: str  # Human-readable label.
    probability: float  # Final probability from the selected tier.
    fast_probability: float  # Fast-tier probability, always reported.
    heavy_probability: float | None = None  # Heavy-tier probability when escalated.
    model_used: str  # fast-lite or heavy-pro.
    cascade_triggered: bool  # Whether the heavy tier was consulted.
    latency_ms: float  # Server-side inference time.
    timestamp: str  # UTC ISO-8601 timestamp.


models: dict = {"fast_lite": None, "heavy_pro": None}  # In-process model handles.
model_versions: dict = {}  # Version numbers currently loaded.


def init_db() -> None:  # Create the prediction table if it is missing.
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)  # Ensure the bind-mounted logs directory exists.
    with sqlite3.connect(DB_PATH) as conn:  # Open a short-lived connection.
        conn.execute(  # Schema is append-only and drift-check friendly.
            """
            CREATE TABLE IF NOT EXISTS prediction_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                cpu_utilization REAL, memory_usage REAL, disk_io REAL, network_latency REAL,
                process_count REAL, thread_count REAL, context_switches REAL, cache_miss_rate REAL,
                temperature REAL, power_consumption REAL, uptime REAL,
                prediction INTEGER NOT NULL, probability REAL NOT NULL,
                model_used TEXT NOT NULL, cascade_triggered INTEGER NOT NULL, latency_ms REAL NOT NULL
            )
            """
        )
        conn.commit()  # Make the table visible to other processes.


def load_models() -> None:  # Resolve both @champion models and their versions.
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)  # Point the client at the tracking server.
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)  # Registry client for version metadata.
    for key, name in (("fast_lite", FAST_MODEL_NAME), ("heavy_pro", HEAVY_MODEL_NAME)):  # Load each tier.
        models[key] = mlflow.sklearn.load_model(f"models:/{name}@{MODEL_ALIAS}")  # Pull the champion artifact.
        model_versions[key] = client.get_model_version_by_alias(name, MODEL_ALIAS).version  # Record its version.


@asynccontextmanager
async def lifespan(_: FastAPI):  # Start the DB and best-effort model load.
    init_db()  # Always prepare the log table.
    try:  # Training may not have run yet, so startup must not crash.
        load_models()  # Warm both tiers from MLflow.
        print("[aiops-api] Loaded @champion models from MLflow.")  # Visible in docker logs.
    except Exception as exc:  # Degrade gracefully until /reload is called.
        print(f"[aiops-api] Startup note: models not loaded yet ({exc}).")  # Actionable startup message.
    yield  # Hand control to FastAPI.


app = FastAPI(title="SentinelRoute AIOps Cascade API", version="1.0.0", lifespan=lifespan)  # The serving app.


def log_prediction(data: dict, result: PredictionResponse) -> None:  # Append one prediction row.
    init_db()  # Safe to call on every request.
    cols = ", ".join(FEATURE_COLS)  # Column list for the insert.
    placeholders = ", ".join(["?"] * (len(FEATURE_COLS) + 6))  # Inputs plus timestamp/result fields.
    vals = [result.timestamp] + [data[c] for c in FEATURE_COLS] + [  # Values in schema order.
        result.prediction, result.probability, result.model_used,
        int(result.cascade_triggered), result.latency_ms,
    ]
    with sqlite3.connect(DB_PATH) as conn:  # Short transaction per request.
        conn.execute(  # Parameterised insert avoids SQL injection.
            f"INSERT INTO prediction_logs (timestamp, {cols}, prediction, probability, model_used, cascade_triggered, latency_ms) VALUES ({placeholders})",
            vals,
        )
        conn.commit()  # Persist before responding.


@app.get("/")  # Small landing payload for evaluators opening the base URL.
def root():
    return {"service": "SentinelRoute AIOps API", "docs": "/docs", "health": "/health", "predict": "/predict"}  # Discoverable endpoints.


@app.get("/health")  # Readiness plus registry metadata.
def health(response: Response):
    loaded = {"fast_lite": models["fast_lite"] is not None, "heavy_pro": models["heavy_pro"] is not None}  # Per-tier state.
    if not all(loaded.values()):  # Fail readiness until both tiers exist.
        response.status_code = 503  # Signal "not ready" to orchestrators.
        return {"status": "models_not_loaded", "models_loaded": loaded}  # Explain what is missing.
    return {  # Happy-path payload used by the demo.
        "status": "ok",
        "models_loaded": loaded,
        "alias": MODEL_ALIAS,
        "model_versions": model_versions,
    }


@app.post("/reload")  # Hot-swap to the current champions without a restart.
def reload_models():
    try:  # Surface registry errors as HTTP 503.
        load_models()  # Refresh handles and versions.
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Failed to load models: {exc}")  # Keep the old handles intact.
    return {"status": "reloaded", "fast_lite": FAST_MODEL_NAME, "heavy_pro": HEAVY_MODEL_NAME}  # Confirm success.


@app.post("/predict", response_model=PredictionResponse)  # Cascade inference endpoint.
def predict(payload: TelemetryRequest, response: Response):
    if models["fast_lite"] is None or models["heavy_pro"] is None:  # Refuse traffic before training.
        raise HTTPException(status_code=503, detail="Models not loaded. Train and call POST /reload.")  # Actionable error.

    start = time.perf_counter()  # Start the latency clock.
    data_dict = payload.model_dump()  # Validated fields as a plain dict.
    df = pd.DataFrame([[data_dict[c] for c in FEATURE_COLS]], columns=FEATURE_COLS)  # One-row feature frame.

    p_lite = float(models["fast_lite"].predict_proba(df)[0][1])  # Fast-tier anomaly probability.
    p_heavy = None  # Only computed when the request escalates.
    if p_lite <= CASCADE_LOW or p_lite >= CASCADE_HIGH:  # Confident fast decision.
        final_prob, model_used, cascade_triggered = p_lite, "fast-lite", False  # Stay on the cheap tier.
    else:  # Uncertain band.
        p_heavy = float(models["heavy_pro"].predict_proba(df)[0][1])  # Ask the heavy tier.
        final_prob, model_used, cascade_triggered = p_heavy, "heavy-pro", True  # Record the escalation.

    prediction = 1 if final_prob >= 0.50 else 0  # Fixed decision threshold.
    result = PredictionResponse(  # Build the typed response.
        prediction=prediction,
        status_label="anomaly" if prediction == 1 else "normal",
        probability=round(final_prob, 4),
        fast_probability=round(p_lite, 4),
        heavy_probability=round(p_heavy, 4) if p_heavy is not None else None,
        model_used=model_used,
        cascade_triggered=cascade_triggered,
        latency_ms=round((time.perf_counter() - start) * 1000.0, 2),
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    response.headers["X-Model-Route"] = model_used  # Expose the route for the demo and clients.
    log_prediction(data_dict, result)  # Persist before returning.
    return result  # Send the prediction.


@app.get("/stats")  # Aggregate counters for the demo.
def stats():
    init_db()  # Ensure the table exists even before any prediction.
    with sqlite3.connect(DB_PATH) as conn:  # Read the rolling log.
        row = conn.execute(  # Single aggregate query keeps it cheap.
            """
            SELECT COUNT(*),
                   SUM(CASE WHEN model_used = 'fast-lite' THEN 1 ELSE 0 END),
                   SUM(CASE WHEN model_used = 'heavy-pro' THEN 1 ELSE 0 END),
                   SUM(CASE WHEN prediction = 1 THEN 1 ELSE 0 END),
                   AVG(latency_ms)
            FROM prediction_logs
            """
        ).fetchone()
    total = row[0] or 0  # Empty tables return None.
    fast_cnt, heavy_cnt, anom_cnt = row[1] or 0, row[2] or 0, row[3] or 0  # Null-safe counters.
    return {  # Stable stats payload.
        "total_requests": total,
        "fast_lite_count": fast_cnt,
        "heavy_pro_count": heavy_cnt,
        "heavy_escalation_pct": round((heavy_cnt / total * 100.0) if total else 0.0, 2),
        "anomalies_detected": anom_cnt,
        "avg_latency_ms": round(float(row[4] or 0.0), 2),
    }
