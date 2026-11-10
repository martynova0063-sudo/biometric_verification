from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import and_, case, func, select
from sqlalchemy.orm import Session

from .models import VerificationSession


def _count(condition: Any) -> Any:
    return func.coalesce(func.sum(case((condition, 1), else_=0)), 0)


def calculate_metrics_snapshot(db: Session, window_days: int, now: datetime | None = None) -> dict[str, Any]:
    """Calculate FAR/FRR and APCER/BPCER from operator-labeled sessions."""
    window_end = now or datetime.now(timezone.utc).replace(tzinfo=None)
    if window_end.tzinfo is not None:
        window_end = window_end.astimezone(timezone.utc).replace(tzinfo=None)
    window_start = window_end - timedelta(days=window_days)
    in_window = and_(
        VerificationSession.created_at >= window_start,
        VerificationSession.created_at <= window_end,
    )

    biometric = db.execute(
        select(
            _count(and_(VerificationSession.is_match.is_(True), VerificationSession.ground_truth_match.is_(True))),
            _count(and_(VerificationSession.is_match.is_(True), VerificationSession.ground_truth_match.is_(False))),
            _count(and_(VerificationSession.is_match.is_(False), VerificationSession.ground_truth_match.is_(True))),
            _count(and_(VerificationSession.is_match.is_(False), VerificationSession.ground_truth_match.is_(False))),
        ).where(
            in_window,
            VerificationSession.is_match.is_not(None),
            VerificationSession.ground_truth_match.is_not(None),
        )
    ).one()
    tp, fp, fn, tn = (int(value or 0) for value in biometric)

    liveness = db.execute(
        select(
            _count(and_(VerificationSession.is_live.is_(True), VerificationSession.ground_truth_liveness.is_(False))),
            _count(and_(VerificationSession.ground_truth_liveness.is_(False))),
            _count(and_(VerificationSession.is_live.is_(False), VerificationSession.ground_truth_liveness.is_(True))),
            _count(and_(VerificationSession.ground_truth_liveness.is_(True))),
        ).where(
            in_window,
            VerificationSession.is_live.is_not(None),
            VerificationSession.ground_truth_liveness.is_not(None),
        )
    ).one()
    spoof_accepted, total_spoof, live_rejected, total_live = (int(value or 0) for value in liveness)

    far = fp / (fp + tn) if fp + tn else 0.0
    frr = fn / (fn + tp) if fn + tp else 0.0
    apcer = spoof_accepted / total_spoof if total_spoof else 0.0
    bpcer = live_rejected / total_live if total_live else 0.0
    total_biometric = tp + fp + fn + tn
    total_liveness = total_spoof + total_live

    return {
        "calculation_date": window_end.date(),
        "window_start": window_start,
        "window_end": window_end,
        "far": round(far, 4),
        "frr": round(frr, 4),
        "cer": round((far + frr) / 2, 4),
        "apcer": round(apcer, 4),
        "bpcer": round(bpcer, 4),
        "acer": round((apcer + bpcer) / 2, 4),
        "total_samples": total_biometric,
        "successful_matches": tp + fp,
        "failed_matches": fn + tn,
        "spoof_attempts_detected": total_spoof - spoof_accepted,
        "live_rejected_as_spoof": live_rejected,
        "calculation_details": {
            "biometric": {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "genuine_cases": tp + fn,
                "impostor_cases": fp + tn,
                "labeled_cases": total_biometric,
            },
            "liveness": {
                "spoof_cases": total_spoof,
                "spoof_accepted_as_live": spoof_accepted,
                "spoof_rejected": total_spoof - spoof_accepted,
                "bona_fide_cases": total_live,
                "bona_fide_rejected_as_spoof": live_rejected,
                "bona_fide_accepted": total_live - live_rejected,
                "labeled_cases": total_liveness,
            },
        },
    }
