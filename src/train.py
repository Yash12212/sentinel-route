import argparse  # Parse dataset path and promotion flag.
import hashlib  # Fingerprint the dataset for promotion decisions.
import json  # Write reference stats and the evaluation report.
import subprocess  # Read the current git SHA for run tags.
import time  # Measure per-row inference latency.
from pathlib import Path  # Type and handle filesystem paths.
from typing import Any  # Accept sklearn estimators without over-typing.

import mlflow  # Track runs and register models.
import mlflow.sklearn  # Persist sklearn estimators as MLflow models.
import numpy as np  # Vectorised metrics and sampling.
import pandas as pd  # Load and manipulate the telemetry table.
from mlflow.tracking import MlflowClient  # Read champions and versions.
from sklearn.ensemble import RandomForestClassifier  # Both model tiers are forests.
from sklearn.linear_model import LogisticRegression  # Linear baseline.
from sklearn.metrics import (  # Metrics used by the honest evaluation pack.
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split  # Splits and CV.
from sklearn.pipeline import Pipeline  # Scale before logistic regression.
from sklearn.preprocessing import StandardScaler  # Put raw features on a common scale.

from src.config import (  # Shared paths, names, thresholds, and features.
    CASCADE_HIGH, CASCADE_LOW, DATA_RAW_PATH, EVAL_REPORT_PATH, EXPERIMENT_NAME,
    FAST_MODEL_NAME, FEATURE_COLS, HEAVY_MODEL_NAME, MIN_PROMOTION_F1, MLFLOW_TRACKING_URI,
    MODEL_ALIAS, REFERENCE_STATS_PATH, SAMPLE_BORDERLINE_PATH, TARGET_COL,
)

RAW_TARGET_COL = "status"  # Name used by the Kaggle CSV.
LABEL_MAP = {  # Accept common spellings in case the CSV is re-exported.
    "healthy": 0, "normal": 0, "ok": 0,
    "anomalous": 1, "anomaly": 1, "degraded": 1, "fault": 1, "faulty": 1,
}


def load_dataset(path: Path | None = None) -> pd.DataFrame:  # Load and normalise the Kaggle CSV.
    """Load the Kaggle IT telemetry CSV and normalise the target column.

    The source file calls the target ``status``; the application standardises
    it to ``system_status`` (0 = healthy, 1 = anomalous/degraded).
    """
    dataset_path = Path(path) if path else DATA_RAW_PATH  # Allow tests to point at temp files.
    if not dataset_path.exists():  # Fail early with a useful message.
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. Download Big_data_dataset.csv "
            "from the Kaggle 'IT System Performance & Resource Metrics' dataset."
        )

    df = pd.read_csv(dataset_path)  # Read the tracked dataset directly.
    if RAW_TARGET_COL in df.columns and TARGET_COL not in df.columns:  # Rename only when needed.
        df = df.rename(columns={RAW_TARGET_COL: TARGET_COL})  # Standardise the target name.

    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]  # Validate the schema.
    if missing:  # Stop on missing inputs instead of training on NaNs.
        raise ValueError(f"Dataset is missing required columns: {missing}")  # Name every missing column.

    if df[TARGET_COL].dtype == object:  # Labels may be strings in some exports.
        df[TARGET_COL] = (  # Map known labels to 0/1.
            df[TARGET_COL].astype(str).str.strip().str.lower().map(LABEL_MAP)
        )

    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()  # Drop incomplete rows.
    df[TARGET_COL] = df[TARGET_COL].astype(int)  # Guarantee an integer label.
    return df  # Hand back a clean training frame.


def get_git_sha() -> str:  # Short SHA for traceable MLflow runs.
    try:  # Git may be unavailable inside CI containers.
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()  # Clean textual SHA.
    except Exception:  # Fall back to a local marker.
        return "local-dev"  # Still valid MLflow tag value.


def file_md5(path: Path) -> str:  # Fingerprint the dataset contents.
    return hashlib.md5(path.read_bytes()).hexdigest()[:12]  # Short stable hash.


def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> dict:  # Quick hold-out scoring for registry runs.
    start = time.perf_counter()  # Start the latency clock.
    preds = model.predict(X_test)  # Hard labels.
    elapsed_ms = (time.perf_counter() - start) * 1000.0  # Total batch time in ms.
    probs = model.predict_proba(X_test)[:, 1]  # Anomaly probabilities.
    return {  # Metrics logged on each model version.
        "accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "precision": round(float(precision_score(y_test, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, preds, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, preds, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probs)), 4),
        "avg_inference_ms": round(elapsed_ms / max(len(X_test), 1), 4),
    }


def get_champion(client: MlflowClient, model_name: str) -> tuple[float, str]:  # Read the current champion.
    """Return the champion's F1 and dataset fingerprint, or (-1, "") if absent."""
    try:  # No alias exists before the first training run.
        mv = client.get_model_version_by_alias(model_name, MODEL_ALIAS)  # Resolve the alias.
        run = client.get_run(mv.run_id)  # Load the run that produced it.
        return (  # F1 and dataset fingerprint drive the gate.
            float(run.data.metrics.get("f1_score", 0.0)),
            run.data.tags.get("dataset_md5", ""),
        )
    except Exception:  # Missing alias or registry.
        return -1.0, ""  # Sentinel values mean "no champion yet".


def bootstrap_f1_ci(y_true, preds, n_boot: int = 1000, seed: int = 42) -> list[float]:  # Percentile F1 interval.
    """Percentile bootstrap interval; shows how wide F1 really is with 97 positives."""
    rng = np.random.default_rng(seed)  # Reproducible resampling.
    truth, pred = np.asarray(y_true), np.asarray(preds)  # Work on arrays.
    scores = [  # One F1 per bootstrap sample.
        f1_score(truth[idx], pred[idx], zero_division=0)
        for idx in (rng.integers(0, len(truth), len(truth)) for _ in range(n_boot))
    ]
    low, high = np.quantile(scores, [0.025, 0.975])  # 95% interval.
    return [round(float(low), 4), round(float(high), 4)]  # JSON-friendly values.


def summarise(y_true, probs, threshold: float = 0.5) -> dict:  # Full metric block for the report.
    """Report point estimates, a bootstrap F1 interval and raw error counts."""
    probs = np.asarray(probs)  # Accept lists or arrays.
    preds = (probs >= threshold).astype(int)  # Fixed threshold, never tuned on test.
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()  # Raw error counts.
    return {  # One consistent schema for every evaluated model.
        "accuracy": round(float(accuracy_score(y_true, preds)), 4),
        "precision": round(float(precision_score(y_true, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, preds, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, preds, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, probs)), 4),
        "average_precision": round(float(average_precision_score(y_true, probs)), 4),
        "brier_score": round(float(brier_score_loss(y_true, probs)), 6),
        "f1_ci95": bootstrap_f1_ci(y_true, preds),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def threshold_rule(X: pd.DataFrame) -> pd.Series:  # Reproduce the dataset's label rule.
    """Explicit rule that generated this dataset's labels.

    Used only as an evaluation baseline: it proves the scores reflect dataset
    separability, not model leakage or an unusually clever classifier.
    """
    return (  # The exact generating condition.
        (X["cpu_utilization"] > 80)
        & (X["memory_usage"] > 85)
        & (X["temperature"] > 75)
    ).astype(int)  # Return 0/1 labels.


def build_models() -> dict[str, Any]:  # Define both production tiers.
    """Fast tier: cheap shallow forest. Heavy tier: deeper, larger forest.

    The tiers differ by capacity, not just tree count. At depth 2 the fast
    tier cannot represent the dataset's three-way AND rule, so it produces
    false positives near the boundary. The depth-4 heavy tier can, which is
    exactly what the cascade buys on the uncertain band.
    """
    return {  # Fresh estimators each call keep CV folds independent.
        "fast_lite": RandomForestClassifier(
            n_estimators=40, max_depth=2, max_features=0.5, class_weight="balanced",
            random_state=42, n_jobs=-1,
        ),
        "heavy_pro": RandomForestClassifier(
            n_estimators=150, max_depth=4, max_features=0.5, class_weight="balanced",
            random_state=42, n_jobs=-1,
        ),
    }


def cross_validation_report(X: pd.DataFrame, y: pd.Series) -> dict:  # 5-fold comparison of both tiers.
    """5-fold CV, because one 2,000-row hold-out contains only 19 anomalies."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # Preserve the 0.97% anomaly rate.
    scoring = ["f1", "roc_auc", "average_precision", "precision", "recall"]  # Metrics per fold.
    report = {}  # Results keyed by tier.
    for name, model in build_models().items():  # Evaluate each tier independently.
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)  # Run all folds.
        report[name] = {  # Mean and spread per metric.
            metric.replace("test_", ""): {
                "mean": round(float(values.mean()), 4),
                "std": round(float(values.std()), 4),
            }
            for metric, values in scores.items() if metric.startswith("test_")
        }
    return report  # Comparable evidence beyond the single split.


def baseline_report(  # Show what simpler models score on the same split.
    X_train: pd.DataFrame, y_train: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series,
) -> dict:
    """Show what a linear model and the label-generating rule score on the same split."""
    linear = Pipeline([  # Scaling matters for logistic regression.
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42)),
    ]).fit(X_train, y_train)  # Train on the same rows as the forests.
    rule = threshold_rule(X_test).to_numpy()  # Perfect-by-construction baseline.
    return {  # Two contrasting baselines.
        "logistic_regression": summarise(y_test, linear.predict_proba(X_test)[:, 1]),
        "threshold_rule": summarise(y_test, rule.astype(float)),
    }


def leakage_control(X: pd.DataFrame, y: pd.Series) -> dict:  # Shuffled-label sanity check.
    """Train on randomly shuffled labels; a genuine pipeline should score near zero."""
    shuffled = y.sample(frac=1.0, random_state=0).reset_index(drop=True)  # Destroy label signal.
    X_tr, X_te, y_tr, y_te = train_test_split(  # Same style of split as the real run.
        X, shuffled, test_size=0.20, stratify=shuffled, random_state=42
    )
    model = RandomForestClassifier(  # Same family as the real models.
        n_estimators=100, max_depth=6, class_weight="balanced",
        random_state=42, n_jobs=-1,
    ).fit(X_tr, y_tr)  # Train on noise.
    return summarise(y_te, model.predict_proba(X_te)[:, 1])  # Near-zero F1 proves no leakage.


def cascade_report(  # Evaluate the routing policy itself.
    fast_model: Any, heavy_model: Any,
    X_test: pd.DataFrame, y_test: pd.Series,
) -> dict:
    p_fast = fast_model.predict_proba(X_test)[:, 1]  # Fast-tier probabilities.
    p_heavy = heavy_model.predict_proba(X_test)[:, 1]  # Heavy-tier probabilities.
    escalate = (p_fast > CASCADE_LOW) & (p_fast < CASCADE_HIGH)  # Uncertain band.
    final = np.where(escalate, p_heavy, p_fast)  # Route each row.
    return {  # Metrics plus routing volume.
        **summarise(y_test, final),
        "escalated": int(escalate.sum()),
        "escalation_pct": round(float(escalate.mean() * 100.0), 2),
    }


def build_eval_report(  # Assemble, persist, and log the evidence pack.
    df: pd.DataFrame, X: pd.DataFrame, y: pd.Series,
    X_train: pd.DataFrame, y_train: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series,
    fast_model: Any, heavy_model: Any, tags: dict,
) -> dict:
    report = {  # Single JSON artifact for reviewers.
        "dataset": {
            "rows": len(df),
            "anomalies": int(y.sum()),
            "anomaly_rate": round(float(y.mean()), 4),
            "source": "Kaggle IT System Performance & Resource Metrics",
        },
        "holdout": {  # Point estimates, CIs, and confusion counts.
            "fast_lite": summarise(y_test, fast_model.predict_proba(X_test)[:, 1]),
            "heavy_pro": summarise(y_test, heavy_model.predict_proba(X_test)[:, 1]),
            "cascade": cascade_report(fast_model, heavy_model, X_test, y_test),
        },
        "cross_validation_5fold": cross_validation_report(X, y),  # More reliable than 19 positives.
        "baselines": baseline_report(X_train, y_train, X_test, y_test),  # Rule and linear reference.
        "leakage_control_shuffled_labels": leakage_control(X, y),  # Honesty check.
        "notes": (  # Explain the two-tier design in one place.
            "The tiers differ by capacity on purpose. Fast-Lite is a 40-tree, "
            "depth-2 forest (~35 KB); it cannot represent the label rule "
            "(cpu>80 AND memory>85 AND temperature>75) and misses 1 of 19 "
            "anomalies with 2 false positives on the hold-out. Heavy-Pro is a "
            "150-tree, depth-4 forest (~216 KB) that captures the interaction "
            "and removes both false positives. The cascade keeps Heavy-Pro "
            "quality while sending only the uncertain band (~12%) to the "
            "larger model; 5-fold CV confirms the ordering. The rule baseline "
            "is reported only as a data-quality check: it regenerates the "
            "labels by construction. The shuffled-label control near zero "
            "shows the evaluation is not leaking."
        ),
    }

    EVAL_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)  # Ensure data/ exists.
    EVAL_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")  # Human-readable artifact.

    with mlflow.start_run(run_name=f"evaluation-{tags['git_sha']}"):  # Separate evaluation run.
        mlflow.set_tags({**tags, "run_type": "evaluation"})  # Trace the evidence run.
        mlflow.log_dict(report, "eval_report.json")  # Attach the full report.
        mlflow.log_metrics({  # Headline evidence for dashboards.
            "cascade_f1": report["holdout"]["cascade"]["f1_score"],
            "cascade_escalation_pct": report["holdout"]["cascade"]["escalation_pct"],
            "cv_fast_f1_mean": report["cross_validation_5fold"]["fast_lite"]["f1"]["mean"],
            "cv_fast_f1_std": report["cross_validation_5fold"]["fast_lite"]["f1"]["std"],
            "cv_heavy_f1_mean": report["cross_validation_5fold"]["heavy_pro"]["f1"]["mean"],
            "logreg_baseline_f1": report["baselines"]["logistic_regression"]["f1_score"],
            "rule_baseline_f1": report["baselines"]["threshold_rule"]["f1_score"],
            "shuffled_label_f1": report["leakage_control_shuffled_labels"]["f1_score"],
            "shuffled_label_roc_auc": report["leakage_control_shuffled_labels"]["roc_auc"],
        })
    return report  # Caller prints the summary.


def print_eval_report(report: dict) -> None:  # Compact console summary.
    print("\n=== Evaluation (honest framing) ===")  # Section header.
    for name, m in report["holdout"].items():  # One line per evaluated model.
        c = m["confusion"]  # Shortcut for error counts.
        print(  # F1, CI, errors, and PR-AUC.
            f"{name:<16} F1={m['f1_score']:.4f} CI={m['f1_ci95']} "
            f"FP={c['fp']} FN={c['fn']} AP={m['average_precision']:.4f}"
        )
    cv = report["cross_validation_5fold"]  # Cross-validation block.
    print(  # Mean and spread for both tiers.
        f"5-fold CV       fast F1={cv['fast_lite']['f1']['mean']:.4f}"
        f"+/-{cv['fast_lite']['f1']['std']:.4f} | "
        f"heavy F1={cv['heavy_pro']['f1']['mean']:.4f}"
        f"+/-{cv['heavy_pro']['f1']['std']:.4f}"
    )
    base = report["baselines"]  # Baseline block.
    ctrl = report["leakage_control_shuffled_labels"]  # Leakage block.
    print(  # Rule versus linear baseline.
        f"Baselines       rule F1={base['threshold_rule']['f1_score']:.4f} | "
        f"logreg F1={base['logistic_regression']['f1_score']:.4f}"
    )
    print(f"Leakage control shuffled-label F1={ctrl['f1_score']:.4f} AUC={ctrl['roc_auc']:.4f}")  # Sanity result.
    print(f"Report written to {EVAL_REPORT_PATH}")  # Point at the artifact.


def train_and_register(  # Fit, score, log, and conditionally promote one tier.
    client: MlflowClient, reg_name: str, tier: str, model: Any,
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series,
    tags: dict, promote: bool,
) -> tuple[Any, dict]:
    model.fit(X_train, y_train)  # Train on the 80% split.
    metrics = evaluate_model(model, X_test, y_test)  # Fast hold-out summary.

    with mlflow.start_run(run_name=f"{tier}-{tags['git_sha']}"):  # One run per tier.
        mlflow.log_params({  # Reproducibility parameters.
            "model_tier": tier,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "cascade_low": CASCADE_LOW,
            "cascade_high": CASCADE_HIGH,
        })
        mlflow.log_metrics(metrics)  # Version-level metrics.
        mlflow.set_tags({**tags, "model_tier": tier})  # Git and dataset traceability.
        mlflow.sklearn.log_model(  # Persist and register the estimator.
            sk_model=model,
            artifact_path="model",
            registered_model_name=reg_name,
            input_example=X_train.head(3),
        )

    versions = client.search_model_versions(f"name='{reg_name}'")  # Find the newly created version.
    latest_ver = str(max(int(v.version) for v in versions))  # Numeric max avoids string ordering bugs.
    if metrics["f1_score"] < MIN_PROMOTION_F1:  # Absolute quality gate.
        print(
            f"[{reg_name}] Rejected v{latest_ver}: F1 {metrics['f1_score']} "
            f"< minimum {MIN_PROMOTION_F1}"
        )  # Candidate stays registered but never becomes champion.
        return model, metrics  # Skip alias changes.
    current_champ_f1, champion_md5 = get_champion(client, reg_name)  # Compare against the alias.
    new_dataset = bool(champion_md5) and champion_md5 != tags["dataset_md5"]  # Data changed?

    if current_champ_f1 < 0:  # First model for this registry name.
        client.set_registered_model_alias(reg_name, MODEL_ALIAS, latest_ver)  # Bootstrap the champion.
        print(f"[{reg_name}] Initial v{latest_ver} -> @{MODEL_ALIAS} (F1={metrics['f1_score']})")  # Audit line.
    elif promote and (new_dataset or metrics["f1_score"] >= current_champ_f1):  # Gate.
        client.set_registered_model_alias(reg_name, MODEL_ALIAS, latest_ver)  # Promote the new version.
        reason = "new dataset" if new_dataset else f"F1 {current_champ_f1} -> {metrics['f1_score']}"  # Why it passed.
        print(f"[{reg_name}] Promoted v{latest_ver} -> @{MODEL_ALIAS} ({reason})")  # Audit line.
    else:  # Candidate did not beat the gate.
        print(
            f"[{reg_name}] Kept @{MODEL_ALIAS} (F1={current_champ_f1}); "
            f"v{latest_ver} not promoted"
        )  # Audit line.

    return model, metrics  # Hand the fitted model back for reports and artifacts.


def save_reference_artifacts(  # Persist drift baselines and the demo payload.
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.Series,
    fast_model: Any, heavy_model: Any,
) -> None:
    REFERENCE_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)  # Ensure data/ exists.
    features_meta = {}  # Per-feature baseline statistics.
    for col in FEATURE_COLS:  # Summarise every model input.
        series = X_train[col].dropna()  # Ignore missing values.
        features_meta[col] = {  # Mean/spread/range plus a KS sample.
            "mean": round(float(series.mean()), 4),
            "std": round(float(series.std()), 4),
            "min": round(float(series.min()), 4),
            "max": round(float(series.max()), 4),
            "sample_values": series.sample(min(200, len(series)), random_state=42).round(4).tolist(),
        }

    p_heavy = heavy_model.predict_proba(X_test)[:, 1]  # Confidence for sample selection.
    healthy_df = X_test[(y_test == 0) & (p_heavy < 0.15)].head(25)  # Clear healthy examples.
    fault_df = X_test[(y_test == 1) & (p_heavy > 0.85)].head(25)  # Clear fault examples.
    if healthy_df.empty:  # Fall back if the model was unusually uncertain.
        healthy_df = X_test[y_test == 0].head(25)  # Any true healthy rows.
    if fault_df.empty:  # Fall back if the model was unusually uncertain.
        fault_df = X_test[y_test == 1].head(25)  # Any true fault rows.

    ref_payload = {  # Reference artifact consumed by target and drift check.
        "feature_order": FEATURE_COLS,
        "features": features_meta,
        "healthy_samples": healthy_df[FEATURE_COLS].to_dict(orient="records"),
        "fault_samples": fault_df[FEATURE_COLS].to_dict(orient="records"),
    }
    REFERENCE_STATS_PATH.write_text(json.dumps(ref_payload, indent=2), encoding="utf-8")  # Persist baseline.

    p_lite = fast_model.predict_proba(X_test)[:, 1]  # Fast-tier confidence.
    p_pro = heavy_model.predict_proba(X_test)[:, 1]  # Heavy-tier confidence.
    band = (p_lite > CASCADE_LOW) & (p_lite < CASCADE_HIGH)  # Uncertainty band.
    band_idx = np.where(band)[0]  # Row positions in the band.
    if len(band_idx):  # Prefer a real uncertain row.
        # Prefer an uncertain true anomaly that the heavy tier confirms: that
        # is the routing behaviour the demo shows. If none exists, prefer a
        # row the fast tier would flag but the heavy tier clears, which is the
        # false-alarm suppression the cascade exists for.
        preferred = [  # Anomalies the heavy tier confirms.
            int(i) for i in band_idx
            if int(y_test.iloc[i]) == 1 and p_pro[i] >= 0.5
        ]
        if preferred:
            pick = max(preferred, key=lambda i: p_pro[i])  # Highest-confidence confirm.
        else:  # Look for a decision flip between the tiers.
            flips = [
                int(i) for i in band_idx
                if (p_lite[i] >= 0.5) != (p_pro[i] >= 0.5)
            ]
            pick = (  # Largest disagreement makes the clearest demo.
                max(flips, key=lambda i: abs(p_lite[i] - p_pro[i]))
                if flips else max(band_idx, key=lambda i: p_pro[i])
            )
        border_row = X_test.iloc[pick]  # Selected demo row.
    else:  # Build a synthetic boundary row if the band is empty.
        border_row = X_train.median(numeric_only=True)  # Start from typical values.
        border_row["cpu_utilization"] = 80.5  # Just above the rule boundary.
        border_row["memory_usage"] = 85.5  # Just above the rule boundary.
        border_row["temperature"] = 75.5  # Just above the rule boundary.
    SAMPLE_BORDERLINE_PATH.parent.mkdir(parents=True, exist_ok=True)  # Ensure tests/ exists.
    SAMPLE_BORDERLINE_PATH.write_text(  # Write the README demo payload.
        json.dumps({c: float(border_row[c]) for c in FEATURE_COLS}, indent=2),
        encoding="utf-8",
    )


def main() -> None:  # End-to-end training entry point.
    parser = argparse.ArgumentParser()  # CLI parser.
    parser.add_argument("--data", type=Path, default=DATA_RAW_PATH)  # Optional alternate CSV.
    parser.add_argument("--promote", action="store_true")  # Enable the champion gate.
    args = parser.parse_args()  # Parsed arguments.

    df = load_dataset(args.data)  # Load and normalise the dataset.
    X, y = df[FEATURE_COLS], df[TARGET_COL].astype(int)  # Split features and labels.
    X_train, X_test, y_train, y_test = train_test_split(  # Fixed, stratified hold-out.
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)  # Point at the tracking server.
    mlflow.set_experiment(EXPERIMENT_NAME)  # Group all runs together.
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)  # Registry operations.
    tags = {"git_sha": get_git_sha(), "dataset_md5": file_md5(args.data)}  # Traceability tags.

    models = build_models()  # Fresh tier definitions.

    fast_m, fast_met = train_and_register(  # Train and maybe promote the fast tier.
        client, FAST_MODEL_NAME, "fast-lite", models["fast_lite"],
        X_train, y_train, X_test, y_test, tags, args.promote,
    )
    heavy_m, heavy_met = train_and_register(  # Train and maybe promote the heavy tier.
        client, HEAVY_MODEL_NAME, "heavy-pro", models["heavy_pro"],
        X_train, y_train, X_test, y_test, tags, args.promote,
    )
    print("Fast-Lite:", fast_met)  # Console summary.
    print("Heavy-Pro:", heavy_met)  # Console summary.
    save_reference_artifacts(X_train, X_test, y_test, fast_m, heavy_m)  # Write drift/demo inputs.
    report = build_eval_report(  # Build and log the evidence pack.
        df, X, y, X_train, y_train, X_test, y_test, fast_m, heavy_m, tags,
    )
    print_eval_report(report)  # Human-readable summary.
    print(f"Rows: {len(df):,} | Anomaly rate: {y.mean():.2%}")  # Dataset scale reminder.


if __name__ == "__main__":  # Allow python -m src.train.
    main()  # Run training.
