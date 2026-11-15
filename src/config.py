import os  # Read environment overrides without extra dependencies.
from pathlib import Path  # Build OS-independent filesystem paths.

BASE_DIR = Path(__file__).resolve().parent.parent  # Repository root, one level above src/.

DATA_RAW_PATH = Path(os.getenv("DATA_RAW_PATH", BASE_DIR / "data" / "raw" / "Big_data_dataset.csv"))  # Tracked Kaggle CSV.
REFERENCE_STATS_PATH = Path(os.getenv("REFERENCE_STATS_PATH", BASE_DIR / "data" / "reference_stats.json"))  # Training drift baseline.
EVAL_REPORT_PATH = Path(os.getenv("EVAL_REPORT_PATH", BASE_DIR / "data" / "eval_report.json"))  # Honest evaluation evidence.
SAMPLE_BORDERLINE_PATH = Path(os.getenv("SAMPLE_BORDERLINE_PATH", BASE_DIR / "tests" / "sample_borderline.json"))  # Demo payload near the cascade band.
DB_PATH = Path(os.getenv("DB_PATH", BASE_DIR / "logs" / "predictions.db"))  # SQLite prediction log.
HEALING_LOG_PATH = Path(os.getenv("HEALING_LOG_PATH", BASE_DIR / "logs" / "healing_events.log"))  # MTTR audit trail.

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")  # Tracking/registry endpoint.
EXPERIMENT_NAME = "sentinel-route-training"  # MLflow experiment holding all runs.
FAST_MODEL_NAME = "sentinel-fast-lite"  # Registry name for the small forest.
HEAVY_MODEL_NAME = "sentinel-heavy-pro"  # Registry name for the larger forest.
MODEL_ALIAS = "champion"  # Alias the API and gate resolve.
MIN_PROMOTION_F1 = float(os.getenv("MIN_PROMOTION_F1", "0.85"))  # Absolute quality floor before promotion.

CASCADE_LOW = float(os.getenv("CASCADE_LOW", "0.20"))  # Below this the fast tier is trusted.
CASCADE_HIGH = float(os.getenv("CASCADE_HIGH", "0.80"))  # Above this the fast tier is trusted.

POLL_INTERVAL_SEC = int(os.getenv("POLL_INTERVAL_SEC", "5"))  # Watchdog poll cadence.
ANOMALY_STREAK_LIMIT = int(os.getenv("ANOMALY_STREAK_LIMIT", "3"))  # Consecutive anomalies before healing.
HEAL_COOLDOWN_SEC = int(os.getenv("HEAL_COOLDOWN_SEC", "30"))  # Pause after a restart to avoid loops.
MIN_DRIFT_SAMPLES = int(os.getenv("MIN_DRIFT_SAMPLES", "50"))  # Minimum live rows for a drift test.

FEATURE_COLS = [  # Exactly the 11 telemetry inputs used by both models.
    "cpu_utilization", "memory_usage", "disk_io", "network_latency",
    "process_count", "thread_count", "context_switches", "cache_miss_rate",
    "temperature", "power_consumption", "uptime",
]
TARGET_COL = "system_status"  # Normalised 0/1 anomaly label.
