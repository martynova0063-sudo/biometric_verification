"""Per-employee HEALTH_ID and measurement corridors over recent history."""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import HealthResult

PERSONAL_BASELINE_WINDOW_DAYS = 30
PERSONAL_BASELINE_MINIMUM_SAMPLES = 4

METRICS = (
    {
        "key": "health_id",
        "label": "HEALTH_ID",
        "unit": "балл 0–1",
        "minimum": 0.0,
        "maximum": 1.0,
    },
    {
        "key": "sleep_hours",
        "label": "Сон",
        "unit": "ч",
        "minimum": 0.0,
        "maximum": 24.0,
    },
    {
        "key": "stress_score",
        "label": "Стресс",
        "unit": "балл",
        "minimum": 0.0,
        "maximum": 10.0,
    },
    {
        "key": "activity_minutes",
        "label": "Активность",
        "unit": "мин",
        "minimum": 0.0,
        "maximum": None,
    },
    {
        "key": "resting_pulse",
        "label": "Пульс в покое",
        "unit": "уд/мин",
        "minimum": 20.0,
        "maximum": 240.0,
    },
    {
        "key": "respiratory_rate",
        "label": "Частота дыхания",
        "unit": "вдох/мин",
        "minimum": 4.0,
        "maximum": 60.0,
    },
)


def _quantile(sorted_values: list[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return sorted_values[lower_index]
    fraction = position - lower_index
    return (
        sorted_values[lower_index] * (1 - fraction)
        + sorted_values[upper_index] * fraction
    )


def _metric_corridor(
    values: list[float],
    *,
    minimum: float,
    maximum: float | None,
) -> dict[str, float | int | bool | None]:
    ordered = sorted(values)
    sample_count = len(ordered)
    ready = sample_count >= PERSONAL_BASELINE_MINIMUM_SAMPLES
    if not sample_count:
        median = None
    else:
        median = _quantile(ordered, 0.5)

    if not ready:
        return {
            "sample_count": sample_count,
            "ready": False,
            "median": median,
            "iqr": None,
            "lower_bound": None,
            "upper_bound": None,
        }

    first_quartile = _quantile(ordered, 0.25)
    third_quartile = _quantile(ordered, 0.75)
    iqr = third_quartile - first_quartile
    lower_bound = max(minimum, median - 1.5 * iqr)
    upper_bound = median + 1.5 * iqr
    if maximum is not None:
        upper_bound = min(maximum, upper_bound)
    return {
        "sample_count": sample_count,
        "ready": True,
        "median": median,
        "iqr": iqr,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
    }


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def calculate_personal_baseline(
    db: Session,
    employee_ref: str,
    *,
    now: datetime | None = None,
) -> dict:
    window_end = now or datetime.now(timezone.utc).replace(tzinfo=None)
    if window_end.tzinfo is not None:
        window_end = window_end.astimezone(timezone.utc).replace(tzinfo=None)
    window_start = window_end - timedelta(days=PERSONAL_BASELINE_WINDOW_DAYS)
    results = list(
        db.scalars(
            select(HealthResult)
            .where(
                HealthResult.employee_ref == employee_ref,
                HealthResult.event_date >= window_start,
                HealthResult.event_date <= window_end,
            )
            .order_by(HealthResult.event_date.asc(), HealthResult.id.asc())
        )
    )

    metrics = []
    for metric in METRICS:
        key = metric["key"]
        values = []
        for result in results:
            raw_value = (
                result.health_id
                if key == "health_id"
                else (result.measurements or {}).get(key)
            )
            value = _finite_number(raw_value)
            if value is not None:
                values.append(value)
        metrics.append({
            "key": key,
            "label": metric["label"],
            "unit": metric["unit"],
            **_metric_corridor(
                values,
                minimum=metric["minimum"],
                maximum=metric["maximum"],
            ),
        })

    return {
        "employee_ref": employee_ref,
        "window_start": window_start,
        "window_end": window_end,
        "window_days": PERSONAL_BASELINE_WINDOW_DAYS,
        "minimum_samples": PERSONAL_BASELINE_MINIMUM_SAMPLES,
        "metrics": metrics,
    }


def compare_result_to_personal_baseline(
    result: HealthResult,
    history: list[HealthResult],
) -> dict[str, object]:
    """Compare a result with the employee's history before its event date."""
    window_start = result.event_date - timedelta(
        days=PERSONAL_BASELINE_WINDOW_DAYS
    )
    prior_results = [
        prior
        for prior in history
        if window_start <= prior.event_date < result.event_date
    ]
    outside_metrics = []
    evaluated_metric_count = 0

    for metric in METRICS:
        key = metric["key"]
        values = []
        for prior in prior_results:
            raw_value = (
                prior.health_id
                if key == "health_id"
                else (prior.measurements or {}).get(key)
            )
            value = _finite_number(raw_value)
            if value is not None:
                values.append(value)

        current_value = _finite_number(
            result.health_id
            if key == "health_id"
            else (result.measurements or {}).get(key)
        )
        corridor = _metric_corridor(
            values,
            minimum=metric["minimum"],
            maximum=metric["maximum"],
        )
        if (
            not corridor["ready"]
            or current_value is None
            or corridor["lower_bound"] is None
            or corridor["upper_bound"] is None
        ):
            continue

        evaluated_metric_count += 1
        if (
            current_value < corridor["lower_bound"]
            or current_value > corridor["upper_bound"]
        ):
            outside_metrics.append(key)

    if outside_metrics:
        status = "outside_corridor"
    elif evaluated_metric_count:
        status = "within_corridor"
    else:
        status = "insufficient_data"

    return {
        "status": status,
        "evaluated_metric_count": evaluated_metric_count,
        "total_metric_count": len(METRICS),
        "outside_metrics": outside_metrics,
    }


def evaluate_result_list_baselines(
    db: Session,
    results: list[HealthResult],
) -> dict[str, dict[str, object]]:
    """Batch-load history and annotate list results without exposing measurements."""
    if not results:
        return {}

    employee_refs = {result.employee_ref for result in results}
    earliest_window = min(
        result.event_date for result in results
    ) - timedelta(days=PERSONAL_BASELINE_WINDOW_DAYS)
    latest_event = max(result.event_date for result in results)
    history_rows = list(
        db.scalars(
            select(HealthResult)
            .where(
                HealthResult.employee_ref.in_(employee_refs),
                HealthResult.event_date >= earliest_window,
                HealthResult.event_date < latest_event,
            )
            .order_by(HealthResult.event_date.asc(), HealthResult.id.asc())
        )
    )
    history_by_employee: dict[str, list[HealthResult]] = {}
    for row in history_rows:
        history_by_employee.setdefault(row.employee_ref, []).append(row)

    evaluations = {}
    for result in results:
        evaluations[result.id] = compare_result_to_personal_baseline(
            result,
            history_by_employee.get(result.employee_ref, []),
        )
    return evaluations
