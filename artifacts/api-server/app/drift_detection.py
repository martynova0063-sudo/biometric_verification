"""Daily PSI monitoring for biometric and health-result measurements."""
from __future__ import annotations

import logging
import math
from collections import Counter
from datetime import date, datetime, timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .db import SessionLocal
from .models import (
    BiometricMetricsLog,
    DataDriftBaseline,
    DataDriftControl,
    DataDriftObservation,
    DataDriftRun,
    RegisteredModel,
)
from .threshold_manager import TARGET_FAR, recalibrate_match_threshold

logger = logging.getLogger(__name__)

ANALYSIS_WINDOW_DAYS = 7
BASELINE_DAYS_REQUIRED = 30
MIN_SAMPLES = 100
PSI_BIN_WIDTH = 0.05
PSI_WARNING = 0.10
PSI_CRITICAL = 0.25
ACER_WARNING = 0.05

METRIC_INFO: tuple[dict[str, str], ...] = (
    {"key": "match_score", "label": "Сходство лица", "unit": "оценка 0–1"},
    {"key": "liveness_score", "label": "Живость", "unit": "оценка 0–1"},
    {"key": "quality_score", "label": "Качество захвата", "unit": "оценка 0–1"},
    {"key": "heart_rate", "label": "Частота пульса", "unit": "уд/мин"},
    {"key": "hrv", "label": "Вариабельность ритма", "unit": "мс"},
    {"key": "breathing_rate", "label": "Частота дыхания", "unit": "вдох/мин"},
    {"key": "stress_index", "label": "Индекс стресса", "unit": "оценка 0–10"},
    {"key": "health_id_value", "label": "Значение HEALTH_ID", "unit": "оценка 0–1"},
    {"key": "lighting_brightness", "label": "Яркость кадра", "unit": "уровень 0–255"},
    {"key": "image_sharpness", "label": "Резкость кадра", "unit": "дисперсия"},
)


def utc_now_naive() -> datetime:
    return datetime.utcnow()


def _finite_values(rows: list[DataDriftObservation], metric_key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = (row.metrics or {}).get(metric_key)
        if isinstance(value, bool) or not isinstance(value, (float, int)):
            continue
        numeric = float(value)
        if math.isfinite(numeric):
            values.append(numeric)
    return values


def build_histogram(values: list[float], bin_width: float = PSI_BIN_WIDTH) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for value in values:
        if not math.isfinite(value):
            continue
        bucket = math.floor(value / bin_width)
        key = str(bucket)
        counts[key] = counts.get(key, 0) + 1
    return {"bin_width": bin_width, "sample_count": sum(counts.values()), "counts": counts}


def calculate_psi(
    baseline: dict[str, Any],
    current: dict[str, Any],
    *,
    epsilon: float = 1e-6,
) -> float | None:
    """Compute PSI over the union of observed fixed-width histogram bins."""
    baseline_counts = baseline.get("counts") or {}
    current_counts = current.get("counts") or {}
    baseline_total = int(baseline.get("sample_count") or sum(baseline_counts.values()))
    current_total = int(current.get("sample_count") or sum(current_counts.values()))
    bins = sorted(set(baseline_counts) | set(current_counts))
    if not bins or baseline_total <= 0 or current_total <= 0:
        return None

    # Pseudocounts avoid infinite PSI when a bin occurs only on one side.
    smoothing_total = epsilon * len(bins)
    result = 0.0
    for bucket in bins:
        expected = (float(baseline_counts.get(bucket, 0)) + epsilon) / (baseline_total + smoothing_total)
        actual = (float(current_counts.get(bucket, 0)) + epsilon) / (current_total + smoothing_total)
        result += (actual - expected) * math.log(actual / expected)
    return max(0.0, float(result))


def record_drift_observation(
    db: Session,
    *,
    source_type: str,
    observed_at: datetime,
    is_test: bool,
    metrics: dict[str, Any],
    capture_metadata: dict[str, Any] | None = None,
) -> DataDriftObservation | None:
    cleaned: dict[str, float] = {}
    for key in (item["key"] for item in METRIC_INFO):
        value = metrics.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        numeric = float(value)
        if math.isfinite(numeric):
            cleaned[key] = numeric
    if not cleaned:
        return None
    row = DataDriftObservation(
        source_type=source_type,
        observed_at=observed_at,
        is_test=is_test,
        metrics=cleaned,
        capture_metadata=_clean_capture_metadata(capture_metadata or {}),
    )
    db.add(row)
    return row


def _clean_capture_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    string_fields = {"camera_model": 120, "video_source": 20, "camera_facing_mode": 20}
    numeric_fields = {"camera_width": 8192, "camera_height": 8192, "camera_fps": 240}
    for key, max_length in string_fields.items():
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            cleaned[key] = value.strip()[:max_length]
    for key, upper_bound in numeric_fields.items():
        value = metadata.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
            if 0 <= float(value) <= upper_bound:
                cleaned[key] = float(value)
    return cleaned


def _observations(
    db: Session,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
) -> list[DataDriftObservation]:
    statement = select(DataDriftObservation).where(DataDriftObservation.is_test.is_(False))
    if start is not None:
        statement = statement.where(DataDriftObservation.observed_at >= start)
    if end is not None:
        statement = statement.where(DataDriftObservation.observed_at <= end)
    return list(db.scalars(statement.order_by(DataDriftObservation.observed_at.asc())))


def _baseline(db: Session, now: datetime) -> DataDriftBaseline | None:
    existing = db.get(DataDriftBaseline, 1)
    if existing:
        return existing
    first_observed = db.scalar(
        select(func.min(DataDriftObservation.observed_at)).where(DataDriftObservation.is_test.is_(False))
    )
    if not first_observed or now < first_observed + timedelta(days=BASELINE_DAYS_REQUIRED):
        return None

    rows = _observations(db, start=first_observed, end=now)
    distributions: dict[str, dict[str, Any]] = {}
    for metric in METRIC_INFO:
        values = _finite_values(rows, metric["key"])
        if len(values) >= MIN_SAMPLES:
            distributions[metric["key"]] = build_histogram(values)
    # A baseline without enough primary identity-match observations would give
    # a false sense of coverage, so keep collecting until it is evaluable.
    if "match_score" not in distributions:
        return None

    baseline = DataDriftBaseline(
        id=1,
        window_start=first_observed,
        window_end=now,
        distributions=distributions,
        observation_count=len(rows),
        created_at=now,
    )
    db.add(baseline)
    db.flush()
    return baseline


def _control_state(db: Session) -> DataDriftControl:
    state = db.get(DataDriftControl, 1)
    if not state:
        state = DataDriftControl(id=1, manual_review_mode=False, manual_review_reason=None)
        db.add(state)
        db.flush()
    return state


def _active_match_threshold(db: Session) -> float:
    model = db.scalar(select(RegisteredModel).where(RegisteredModel.status == "current"))
    try:
        value = float((model.parameters or {}).get("match_threshold", 0.85)) if model else 0.85
    except (TypeError, ValueError):
        return 0.85
    return value if math.isfinite(value) else 0.85


def _latest_acer(db: Session) -> float | None:
    row = db.scalar(select(BiometricMetricsLog).order_by(BiometricMetricsLog.created_at.desc()).limit(1))
    if row is None:
        return None
    try:
        value = float(row.acer)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _status_for_psi(psi_values: dict[str, float | None], *, baseline_ready: bool, window_ready: bool) -> str:
    if not baseline_ready or not window_ready:
        return "collecting"
    values = [value for value in psi_values.values() if value is not None]
    if not values:
        return "insufficient_data"
    if any(value >= PSI_CRITICAL for value in values):
        return "critical"
    if any(value > PSI_WARNING for value in values):
        return "warning"
    return "stable"


def _summary_payload(db: Session, now: datetime) -> dict[str, Any]:
    baseline = db.get(DataDriftBaseline, 1)
    latest_run = db.scalar(select(DataDriftRun).order_by(DataDriftRun.calculation_date.desc()).limit(1))
    control = db.get(DataDriftControl, 1)
    first_observed = db.scalar(
        select(func.min(DataDriftObservation.observed_at)).where(DataDriftObservation.is_test.is_(False))
    )
    observations_total = int(db.scalar(
        select(func.count(DataDriftObservation.id)).where(DataDriftObservation.is_test.is_(False))
    ) or 0)
    analysis_start = now - timedelta(days=ANALYSIS_WINDOW_DAYS)
    recent_rows = _observations(db, start=analysis_start, end=now)
    baseline_distributions = baseline.distributions if baseline else {}
    latest_psi = latest_run.psi_values if latest_run else {}
    baseline_rows = (
        _observations(db, start=baseline.window_start, end=baseline.window_end)
        if baseline else []
    )
    baseline_camera_counts = Counter(
        row.capture_metadata.get("camera_model")
        for row in baseline_rows
        if row.capture_metadata and row.capture_metadata.get("camera_model")
    )
    current_camera_counts = Counter(
        row.capture_metadata.get("camera_model")
        for row in recent_rows
        if row.capture_metadata and row.capture_metadata.get("camera_model")
    )
    baseline_brightness = _finite_values(baseline_rows, "lighting_brightness")
    current_brightness = _finite_values(recent_rows, "lighting_brightness")
    metric_rows: list[dict[str, Any]] = []
    for metric in METRIC_INFO:
        key = metric["key"]
        baseline_distribution = baseline_distributions.get(key)
        baseline_count = int(baseline_distribution.get("sample_count", 0)) if baseline_distribution else 0
        current_count = len(_finite_values(recent_rows, key))
        psi = latest_psi.get(key) if baseline and latest_run else None
        if not baseline:
            metric_status = "collecting"
        elif not baseline_distribution:
            metric_status = "insufficient_data"
        elif current_count < MIN_SAMPLES or psi is None:
            metric_status = "insufficient_data"
        elif psi >= PSI_CRITICAL:
            metric_status = "critical"
        elif psi > PSI_WARNING:
            metric_status = "warning"
        else:
            metric_status = "stable"
        metric_rows.append({
            **metric,
            "baseline_samples": baseline_count,
            "current_samples": current_count,
            "psi": psi,
            "status": metric_status,
            "baseline_available": baseline_distribution is not None,
        })

    latest_recalibration = latest_run.recalibration if latest_run else {}
    days_elapsed = 0
    if first_observed:
        days_elapsed = max(0, min(
            BASELINE_DAYS_REQUIRED,
            (now - first_observed).days,
        ))
    if baseline:
        status = latest_run.status if latest_run else "collecting"
        baseline_status = "ready"
    else:
        status = "collecting"
        baseline_status = "collecting"
    return {
        "status": status,
        "baseline_status": baseline_status,
        "manual_review_mode": bool(control.manual_review_mode) if control else False,
        "manual_review_reason": control.manual_review_reason if control else None,
        "needs_recalibration": bool(latest_run.needs_recalibration) if latest_run else False,
        "collection_started_at": first_observed,
        "baseline_window_start": baseline.window_start if baseline else None,
        "baseline_window_end": baseline.window_end if baseline else None,
        "baseline_created_at": baseline.created_at if baseline else None,
        "baseline_camera_models": [
            {"camera_model": model, "samples": count}
            for model, count in baseline_camera_counts.most_common(8)
        ],
        "current_camera_models": [
            {"camera_model": model, "samples": count}
            for model, count in current_camera_counts.most_common(8)
        ],
        "baseline_mean_brightness": (
            sum(baseline_brightness) / len(baseline_brightness) if baseline_brightness else None
        ),
        "current_mean_brightness": (
            sum(current_brightness) / len(current_brightness) if current_brightness else None
        ),
        "observations_total": observations_total,
        "baseline_days_elapsed": days_elapsed,
        "baseline_days_required": BASELINE_DAYS_REQUIRED,
        "analysis_window_start": analysis_start,
        "analysis_window_end": now,
        "minimum_samples": MIN_SAMPLES,
        "target_far": TARGET_FAR,
        "current_match_threshold": _active_match_threshold(db),
        "recalibration_status": latest_recalibration.get("status", "not_run"),
        "proposed_match_threshold": latest_recalibration.get("proposed_threshold"),
        "achieved_far": latest_recalibration.get("achieved_far"),
        "achieved_frr": latest_recalibration.get("achieved_frr"),
        "labelled_genuine_samples": int(latest_recalibration.get("genuine_samples", 0)),
        "labelled_impostor_samples": int(latest_recalibration.get("impostor_samples", 0)),
        "last_run_at": latest_run.created_at if latest_run else None,
        "alert_messages": latest_run.alert_messages if latest_run else [],
        "metrics": metric_rows,
    }


def get_data_drift_summary(db: Session, now: datetime | None = None) -> dict[str, Any]:
    return _summary_payload(db, now or utc_now_naive())


def run_data_drift_analysis(db: Session, now: datetime | None = None) -> dict[str, Any]:
    now = now or utc_now_naive()
    today: date = now.date()
    baseline = _baseline(db, now)
    control = _control_state(db)
    current_window_start = now - timedelta(days=ANALYSIS_WINDOW_DAYS)
    analysis_ready = bool(baseline and now >= baseline.window_end + timedelta(days=ANALYSIS_WINDOW_DAYS))
    current_start = max(current_window_start, baseline.window_end) if baseline else current_window_start
    query_start = current_start + timedelta(microseconds=1) if baseline else current_start
    current_rows = _observations(db, start=query_start, end=now)
    baseline_distributions = baseline.distributions if baseline else {}
    baseline_rows = (
        _observations(db, start=baseline.window_start, end=baseline.window_end)
        if baseline else []
    )
    baseline_camera_models = {
        row.capture_metadata.get("camera_model")
        for row in baseline_rows
        if row.capture_metadata and row.capture_metadata.get("camera_model")
    }
    current_camera_models = {
        row.capture_metadata.get("camera_model")
        for row in current_rows
        if row.capture_metadata and row.capture_metadata.get("camera_model")
    }
    newly_observed_cameras = sorted(current_camera_models - baseline_camera_models)

    psi_values: dict[str, float | None] = {}
    sample_counts: dict[str, int] = {}
    for metric in METRIC_INFO:
        key = metric["key"]
        current_values = _finite_values(current_rows, key)
        sample_counts[key] = len(current_values)
        baseline_distribution = baseline_distributions.get(key)
        if analysis_ready and baseline_distribution and len(current_values) >= MIN_SAMPLES:
            psi_values[key] = calculate_psi(baseline_distribution, build_histogram(current_values))
        else:
            psi_values[key] = None

    status = _status_for_psi(psi_values, baseline_ready=baseline is not None, window_ready=analysis_ready)
    alerts: list[str] = []
    for metric in METRIC_INFO:
        value = psi_values[metric["key"]]
        if value is None or value <= PSI_WARNING:
            continue
        severity = "критический" if value >= PSI_CRITICAL else "предупреждение"
        alerts.append(f"{severity}: PSI {metric['label']} = {value:.3f}.")
    if newly_observed_cameras:
        alerts.extend(
            f"Обнаружена камера, отсутствующая в базовой выборке: {camera}."
            for camera in newly_observed_cameras
        )
        if status in {"stable", "insufficient_data"}:
            status = "warning"

    acer = _latest_acer(db)
    if acer is not None and acer > ACER_WARNING:
        alerts.append(f"ACER превышает 5%: {acer:.3%}.")
    needs_recalibration = any(value is not None and value > PSI_WARNING for value in psi_values.values())
    needs_recalibration = needs_recalibration or (acer is not None and acer > ACER_WARNING)
    needs_recalibration = needs_recalibration or bool(newly_observed_cameras)
    if status == "critical":
        control.manual_review_mode = True
        critical_metrics = [
            metric["label"] for metric in METRIC_INFO
            if psi_values[metric["key"]] is not None and psi_values[metric["key"]] >= PSI_CRITICAL
        ]
        control.manual_review_reason = (
            "Критический дрейф PSI: " + ", ".join(critical_metrics)
            + ". Ручной режим необходимо отключить оператору после проверки."
        )

    first_observed = db.scalar(
        select(func.min(DataDriftObservation.observed_at)).where(DataDriftObservation.is_test.is_(False))
    )
    allow_recalibration = bool(
        first_observed and now >= first_observed + timedelta(days=14)
    )
    recalibration = recalibrate_match_threshold(
        db,
        now=now,
        target_far=TARGET_FAR,
        allow_apply=allow_recalibration,
    )
    run = db.scalar(select(DataDriftRun).where(DataDriftRun.calculation_date == today))
    if run is None:
        run = DataDriftRun(
            calculation_date=today,
            window_start=current_start,
            window_end=now,
            status=status,
            psi_values=psi_values,
            sample_counts=sample_counts,
            needs_recalibration=needs_recalibration,
            manual_review_mode=control.manual_review_mode,
            alert_messages=alerts,
            recalibration=recalibration,
            created_at=now,
        )
        db.add(run)
    else:
        run.window_start = current_start
        run.window_end = now
        run.status = status
        run.psi_values = psi_values
        run.sample_counts = sample_counts
        run.needs_recalibration = needs_recalibration
        run.manual_review_mode = control.manual_review_mode
        run.alert_messages = alerts
        run.recalibration = recalibration
        run.created_at = now
    db.commit()
    return _summary_payload(db, now)


def list_data_drift_runs(db: Session, limit: int = 30) -> list[DataDriftRun]:
    return list(
        db.scalars(
            select(DataDriftRun)
            .order_by(DataDriftRun.calculation_date.desc())
            .limit(max(1, min(limit, 100)))
        )
    )


def update_manual_review_mode(
    db: Session,
    *,
    enabled: bool,
    reviewer_note: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    now = now or utc_now_naive()
    latest = db.scalar(select(DataDriftRun).order_by(DataDriftRun.calculation_date.desc()).limit(1))
    if not enabled and latest and latest.status == "critical":
        raise ValueError("При критическом PSI отключить ручной режим нельзя.")
    control = _control_state(db)
    control.manual_review_mode = enabled
    control.manual_review_reason = reviewer_note.strip()[:500] if enabled else None
    db.commit()
    return _summary_payload(db, now)


def run_scheduled_drift_analysis() -> None:
    """Run once on service startup and then from the API's daily worker."""
    with SessionLocal() as db:
        run_data_drift_analysis(db)
