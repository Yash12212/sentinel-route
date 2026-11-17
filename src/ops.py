"""Operational demo and self-healing loops.

Serves the simulated target app and runs the watchdog and drift checks:
    uvicorn src.ops:target_app --host 0.0.0.0 --port 8001
    python -m src.ops watchdog
    python -m src.ops drift [--simulate] [--rows 60]
"""

import argparse  # Subcommand parsing for watchdog and drift.
import json  # Read reference distribution samples.
import os  # Read deployment endpoints from the environment.
import random  # Build shifted traffic and jitter target telemetry.
import shlex  # Split the configurable restart command safely.
import sqlite3  # Read live predictions from the API log.
import subprocess  # Restart containers and trigger training.
import sys  # Reuse the current interpreter for retraining.
import time  # Polling cadence and MTTR measurement.
from datetime import datetime  # Timestamp healing events.

import pandas as pd  # Load live rows for the KS test.
import requests  # Talk to the target and API services.
from fastapi import FastAPI  # Simulated target service used by the healing demo.
from scipy.stats import ks_2samp  # Two-sample drift test.

from src.config import (  # Shared operational thresholds and paths.
    ANOMALY_STREAK_LIMIT, DB_PATH, FEATURE_COLS, HEAL_COOLDOWN_SEC,
    HEALING_LOG_PATH, MIN_DRIFT_SAMPLES, POLL_INTERVAL_SEC,
    REFERENCE_STATS_PATH,
)

TARGET_BASE = os.getenv("TARGET_BASE", "http://localhost:8001").rstrip("/")  # Victim service.
API_BASE = os.getenv("API_BASE", "http://localhost:8000").rstrip("/")  # Cascade API base URL.
API_PREDICT = f"{API_BASE}/predict"  # Prediction endpoint used by the watchdog.
API_RELOAD = f"{API_BASE}/reload"  # Hot-reload endpoint used after retraining.
HEAL_RESTART_CMD = os.getenv(  # Deployment-specific restart command.
    "HEAL_RESTART_CMD", "docker compose restart target-service"
)

# --- Simulated target service ------------------------------------------------

target_app = FastAPI(title="Target Microservice (Simulated Production App)", version="1.0.0")  # Demo victim.
FAULT_ACTIVE = False  # Process-local switch cleared whenever the container restarts.

FALLBACK_HEALTHY = {  # Deterministic healthy row if no artifacts exist yet.
    "cpu_utilization": 28.0, "memory_usage": 36.0, "disk_io": 12.0, "network_latency": 22.0,
    "process_count": 180, "thread_count": 620, "context_switches": 450, "cache_miss_rate": 0.03,
    "temperature": 44.0, "power_consumption": 110.0, "uptime": 180.0,
}
FALLBACK_FAULT = {  # Deterministic anomalous row if no artifacts exist yet.
    "cpu_utilization": 96.5, "memory_usage": 94.0, "disk_io": 46.0, "network_latency": 182.0,
    "process_count": 910, "thread_count": 4200, "context_switches": 1880, "cache_miss_rate": 0.18,
    "temperature": 88.5, "power_consumption": 275.0, "uptime": 181.0,
}


def pick_sample(is_fault: bool) -> dict:  # Return one realistic telemetry row.
    if REFERENCE_STATS_PATH.exists():  # Prefer samples captured from the real dataset.
        data = json.loads(REFERENCE_STATS_PATH.read_text(encoding="utf-8"))  # Load healthy/fault pools.
        pool = data["fault_samples"] if is_fault else data["healthy_samples"]  # Choose the matching pool.
        if pool:  # Guard against an empty generated pool.
            row = dict(random.choice(pool))  # Copy a sample so we can perturb it.
            row["cpu_utilization"] = min(99.9, max(1.0, round(row["cpu_utilization"] * random.uniform(0.98, 1.02), 2)))  # Add slight jitter.
            return {k: row[k] for k in FEATURE_COLS}  # Return only model inputs, in order.
    return dict(FALLBACK_FAULT if is_fault else FALLBACK_HEALTHY)  # Fall back to constants before training.


@target_app.get("/health")  # Liveness plus the current fault flag.
def target_health():
    return {"status": "degraded" if FAULT_ACTIVE else "healthy", "fault_active": FAULT_ACTIVE}  # Watchdog reads this.


@target_app.get("/telemetry")  # One row of live-looking telemetry.
def target_telemetry():
    return pick_sample(FAULT_ACTIVE)  # Faults change the emitted distribution.


@target_app.post("/inject-fault")  # Start emitting anomalous telemetry.
def inject_fault():
    global FAULT_ACTIVE  # Mutate the process-local switch.
    FAULT_ACTIVE = True  # Stay faulty until a restart clears it.
    return {"fault_active": True, "message": "Fault injected. Restart container to clear."}  # Confirm the action.


@target_app.post("/clear-fault")  # Manually clear the fault without a restart.
def clear_fault():
    global FAULT_ACTIVE  # Mutate the process-local switch.
    FAULT_ACTIVE = False  # Return to healthy telemetry.
    return {"fault_active": False}  # Confirm the action.


# --- Self-healing and drift loops --------------------------------------------


def wait_until_healthy(timeout_sec: int = 25) -> bool:  # Block until the restarted target is healthy.
    deadline = time.time() + timeout_sec  # Hard upper bound on the wait.
    while time.time() < deadline:  # Poll until success or timeout.
        try:  # Health may refuse connections during a restart.
            r = requests.get(f"{TARGET_BASE}/health", timeout=2)  # Ask the target for its state.
            if r.status_code == 200 and not r.json().get("fault_active", True):  # Healthy means no active fault.
                return True  # Recovery confirmed.
        except Exception:  # Connection errors are expected mid-restart.
            pass  # Keep polling until the deadline.
        time.sleep(1)  # Avoid a tight loop.
    return False  # Recovery did not happen in time.


def run_watchdog() -> None:  # Continuous anomaly detection and self-healing loop.
    streak, first_fault_ts = 0, None  # Track consecutive anomalies and when the streak began.
    HEALING_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)  # Ensure the log directory exists.
    print(f"[watchdog] Polling {TARGET_BASE}/telemetry every {POLL_INTERVAL_SEC}s...")  # Operator hint.

    while True:  # Run until interrupted.
        try:  # One poll must never kill the loop.
            telemetry = requests.get(f"{TARGET_BASE}/telemetry", timeout=4).json()  # Pull live telemetry.
            pred_resp = requests.post(API_PREDICT, json=telemetry, timeout=4)  # Classify it via the cascade API.
            if pred_resp.status_code != 200:  # API may still be starting.
                print(f"[watchdog] API unavailable ({pred_resp.status_code})")  # Surface the condition.
                time.sleep(POLL_INTERVAL_SEC)  # Wait before retrying.
                continue  # Skip streak handling this round.

            res = pred_resp.json()  # Parsed prediction payload.
            now_str = datetime.now().strftime("%H:%M:%S")  # Compact local timestamp for logs.
            if res["prediction"] == 1:  # Anomaly observed.
                if streak == 0:  # First anomaly in a new streak.
                    first_fault_ts = time.time()  # Start the MTTR clock.
                streak += 1  # Extend the streak.
            else:  # Healthy prediction.
                streak, first_fault_ts = 0, None  # Reset the streak.

            print(  # One-line status for the demo.
                f"[{now_str}] route={res['model_used']:<9} | "
                f"pred={res['prediction']} (prob={res['probability']:.2f}) | "
                f"streak={streak}/{ANOMALY_STREAK_LIMIT}"
            )

            if streak >= ANOMALY_STREAK_LIMIT:  # Sustained anomaly.
                print("[ALERT] Sustained anomaly detected. Auto-restarting target-service...")  # Announce healing.
                subprocess.run(shlex.split(HEAL_RESTART_CMD), check=True)  # Restart the victim.
                recovered = wait_until_healthy()  # Wait for real recovery.
                mttr = round(time.time() - (first_fault_ts or time.time()), 2)  # Seconds from fault to recovery.
                with HEALING_LOG_PATH.open("a", encoding="utf-8") as f:  # Append the audit record.
                    f.write(f"{datetime.now().isoformat()} | healed={recovered} | mttr_sec={mttr}\n")  # One line per event.
                print(f"[HEALED] recovered={recovered} | MTTR={mttr}s")  # Console evidence.
                streak, first_fault_ts = 0, None  # Reset after healing.
                time.sleep(HEAL_COOLDOWN_SEC)  # Cool down to avoid restart loops.
                continue  # Start polling again.
        except KeyboardInterrupt:  # Clean operator stop.
            print("\n[watchdog] Stopped.")  # Acknowledge shutdown.
            break  # Exit the loop.
        except Exception as exc:  # Network or transient errors.
            print(f"[watchdog] Poll error: {exc}")  # Keep the loop alive.
        time.sleep(POLL_INTERVAL_SEC)  # Normal poll cadence.


def trigger_retrain() -> None:  # Retrain, gate promotion, and hot-reload the API.
    print("[drift_check] Running local retrain & promotion gate...")  # Operator hint.
    subprocess.run([sys.executable, "-m", "src.train", "--promote"], check=True)  # Run the real trainer.
    try:  # Reload is best-effort if the API is not running.
        r = requests.post(API_RELOAD, timeout=5)  # Swap in the new champions.
        print(f"[drift_check] API /reload -> {r.status_code} {r.json()}")  # Show the result.
    except Exception as exc:  # Local retrain still succeeds without the API.
        print(f"[drift_check] Reload warning: {exc}")  # Report but do not fail.


def simulate_drift(rows: int = 60) -> None:  # Generate shifted traffic for the demo.
    ref = json.loads(REFERENCE_STATS_PATH.read_text(encoding="utf-8"))  # Load the training baseline.
    base_rows = ref["healthy_samples"]  # Start from plausible healthy rows.
    print(f"[drift_check] Sending {rows} shifted telemetry payloads to /predict...")  # Operator hint.
    for _ in range(rows):  # Send a batch large enough for the KS test.
        row = dict(random.choice(base_rows))  # Copy a base row.
        row["cpu_utilization"] = min(99.9, round(row["cpu_utilization"] + 38.0, 2))  # Push CPU upward.
        row["network_latency"] = min(199.9, round(row["network_latency"] + 70.0, 2))  # Push latency upward.
        row["temperature"] = min(94.9, round(row["temperature"] + 18.0, 2))  # Push temperature upward.
        row["power_consumption"] = min(299.9, round(row["power_consumption"] + 65.0, 2))  # Push power upward.
        requests.post(  # Log the shifted sample as a live prediction.
            API_PREDICT,
            json={k: row[k] for k in FEATURE_COLS},
            timeout=5,
        )
    print("[drift_check] Simulated traffic sent. Re-run without --simulate to detect drift.")  # Next step.


def evaluate_drift(live_df: pd.DataFrame, ref_features: dict) -> list[tuple[str, float, float]]:  # Per-feature KS tests.
    """Run a two-sample KS test per feature and return (name, stat, p-value)."""
    results = []  # Collect one tuple per feature.
    for col in FEATURE_COLS:  # Test every model input.
        stat, p_val = ks_2samp(ref_features[col]["sample_values"], live_df[col].dropna().tolist())  # Baseline vs live.
        results.append((col, float(stat), float(p_val)))  # Store rounded-free floats.
    return results  # Caller decides the significance threshold.


def run_drift(simulate: bool, rows: int) -> None:  # Detect drift and retrain when enough features move.
    if simulate:  # Demo path only sends traffic.
        simulate_drift(rows)  # Populate the log with shifted rows.
        return  # Detection runs on the next invocation.

    if not DB_PATH.exists() or not REFERENCE_STATS_PATH.exists():  # Both inputs are required.
        print("[drift_check] Missing prediction DB or reference_stats.json.")  # Actionable message.
        return  # Nothing to do.

    # Open read-only: the Dockerized API writes this DB as root, and we only read here.
    # as_uri() keeps this working on both Linux and Windows.
    with sqlite3.connect(f"{DB_PATH.resolve().as_uri()}?mode=ro", uri=True) as conn:  # Read-only connection.
        live_df = pd.read_sql_query(  # Most recent predictions represent current traffic.
            f"SELECT {', '.join(FEATURE_COLS)} FROM prediction_logs ORDER BY id DESC LIMIT 200",
            conn,
        )

    if len(live_df) < MIN_DRIFT_SAMPLES:  # Too little data for a stable test.
        print(f"[drift_check] Need >= {MIN_DRIFT_SAMPLES} rows (have {len(live_df)}).")  # Tell the operator.
        return  # Skip this run.

    ref_data = json.loads(REFERENCE_STATS_PATH.read_text(encoding="utf-8"))["features"]  # Training baseline.
    drifted_count = 0  # Count features that moved.
    print(f"\n{'Feature':<22} | {'KS Stat':<8} | {'p-value':<10} | {'Drifted?'}")  # Table header.
    print("-" * 58)  # Separator.
    for col, stat, p_val in evaluate_drift(live_df, ref_data):  # Evaluate every feature.
        is_drifted = p_val < 0.05  # Standard significance threshold.
        drifted_count += int(is_drifted)  # Increment the counter.
        print(f"{col:<22} | {stat:<8.4f} | {p_val:<10.6f} | {'YES' if is_drifted else 'no'}")  # One row per feature.

    print(f"\n[drift_check] Drifted features: {drifted_count}/{len(FEATURE_COLS)}")  # Summary line.
    if drifted_count >= 3:  # Enough movement to justify retraining.
        trigger_retrain()  # Retrain and reload.
    else:  # Stable enough.
        print("[drift_check] No retrain needed.")  # Report and stop.


def main() -> None:  # CLI entry point.
    parser = argparse.ArgumentParser(description="SentinelRoute operational loops")  # Root parser.
    sub = parser.add_subparsers(dest="command", required=True)  # Require an explicit mode.

    sub.add_parser("watchdog", help="run the self-healing watchdog loop")  # Healing mode.

    drift = sub.add_parser("drift", help="run the KS drift check and maybe retrain")  # Drift mode.
    drift.add_argument("--simulate", action="store_true", help="send shifted traffic first")  # Demo traffic.
    drift.add_argument("--rows", type=int, default=60)  # Batch size for simulation.

    args = parser.parse_args()  # Parse the chosen mode.
    if args.command == "watchdog":  # Dispatch to healing.
        run_watchdog()  # Run forever until interrupted.
    else:  # Dispatch to drift.
        run_drift(args.simulate, args.rows)  # Simulate and/or detect.


if __name__ == "__main__":  # Allow python -m src.ops.
    main()  # Start the selected loop.
