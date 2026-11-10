"""Conservative, label-backed match-threshold calibration."""
from __future__ import annotations

import math
from datetime import datetime, timedelta

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from .models import ActivityItem, Employee, RegisteredModel, VerificationSession

TARGET_FAR = 0.001
MIN_IMPOSTOR_LABELS = 1_000
MIN_GENUINE_LABELS = 100


def _labelled_scores(db: Session, since: datetime) -> tuple[list[float], list[float]]:
    rows = db.execute(
        select(VerificationSession.match_score, VerificationSession.ground_truth_match)
        .outerjoin(Employee, Employee.personnel_number == VerificationSession.employee_ref)
        .where(
            VerificationSession.created_at >= since,
            VerificationSession.match_score.is_not(None),
            VerificationSession.ground_truth_match.is_not(None),
            or_(Employee.guid.is_(None), Employee.is_test.is_(False)),
        )
    ).all()
    genuine: list[float] = []
    impostor: list[float] = []
    for score, is_genuine in rows:
        numeric_score = float(score)
        if not math.isfinite(numeric_score):
            continue
        (genuine if is_genuine else impostor).append(numeric_score)
    return genuine, impostor


def recalibrate_match_threshold(
    db: Session,
    *,
    now: datetime,
    target_far: float = TARGET_FAR,
    window_days: int = 30,
    allow_apply: bool = True,
) -> dict:
    """Choose the lowest-FRR threshold meeting the empirical FAR target.

    No model setting is changed unless the labelled validation set is large
    enough to make the requested FAR measurable.
    """
    since = now - timedelta(days=window_days)
    genuine, impostor = _labelled_scores(db, since)
    model = db.scalar(select(RegisteredModel).where(RegisteredModel.status == "current"))
    parameters = dict(model.parameters or {}) if model else {}
    try:
        current_threshold = float(parameters.get("match_threshold", 0.85))
    except (TypeError, ValueError):
        current_threshold = 0.85

    base = {
        "target_far": target_far,
        "current_threshold": current_threshold,
        "proposed_threshold": None,
        "achieved_far": None,
        "achieved_frr": None,
        "genuine_samples": len(genuine),
        "impostor_samples": len(impostor),
        "validation_window_days": window_days,
        "status": "insufficient_labels",
    }
    if not allow_apply:
        base["status"] = "initial_collection"
        return base
    if len(impostor) < MIN_IMPOSTOR_LABELS or len(genuine) < MIN_GENUINE_LABELS:
        return base

    candidates = {0.0, 1.0}
    for score in genuine + impostor:
        if 0.0 <= score <= 1.0:
            candidates.add(score)
            # Decisions use score >= threshold. Testing the next representable
            # value also allows a score equal to the maximum impostor to fail.
            just_above = math.nextafter(score, math.inf)
            if just_above <= 1.0:
                candidates.add(just_above)

    feasible: list[tuple[float, float, float]] = []
    for threshold in candidates:
        false_accepts = sum(score >= threshold for score in impostor)
        far = false_accepts / len(impostor)
        if far >= target_far:
            continue
        false_rejects = sum(score < threshold for score in genuine)
        frr = false_rejects / len(genuine)
        feasible.append((frr, threshold, far))

    if not feasible:
        return {**base, "status": "target_unachievable"}

    frr, proposed, far = min(feasible, key=lambda item: (item[0], item[1]))
    base.update({
        "proposed_threshold": proposed,
        "achieved_far": far,
        "achieved_frr": frr,
        "status": "unchanged",
    })
    if not model:
        base["status"] = "model_unavailable"
        return base
    if abs(proposed - current_threshold) < 1e-9:
        return base

    parameters["match_threshold"] = proposed
    model.parameters = parameters
    model.change_summary = (
        f"{model.change_summary.rstrip()} "
        f"Автокалибровка match_threshold={proposed:.6f}: "
        f"эмпирический FAR={far:.6f}, FRR={frr:.6f} "
        f"на {len(impostor)} размеченных impostor и {len(genuine)} genuine."
    )[:2000]
    db.add(ActivityItem(
        type="model",
        title="Порог биометрического совпадения откалиброван",
        description=(
            f"Порог {current_threshold:.4f} → {proposed:.4f}; "
            f"эмпирический FAR {far:.5f}, FRR {frr:.5f}."
        ),
        status="attention",
    ))
    base["status"] = "applied"
    return base
