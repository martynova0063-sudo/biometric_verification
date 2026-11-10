from typing import Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import ReviewTask, VerificationSession


EligibilitySource = Literal["system", "manual"]


def latest_health_id_eligible_verification(
    db: Session,
    employee_ref: str,
) -> tuple[VerificationSession, EligibilitySource, ReviewTask | None] | None:
    """Return the latest qualifying verification, if the employee may receive HEALTH_ID."""
    session = db.scalar(
        select(VerificationSession)
        .where(VerificationSession.employee_ref == employee_ref)
        .order_by(
            VerificationSession.created_at.desc(),
            VerificationSession.id.desc(),
        )
        .limit(1)
    )
    if session is None:
        return None

    latest_review = db.scalar(
        select(ReviewTask)
        .where(
            ReviewTask.kind == "verification",
            ReviewTask.verification_session_id == session.id,
        )
        .order_by(ReviewTask.created_at.desc(), ReviewTask.id.desc())
        .limit(1)
    )

    if latest_review and latest_review.status == "resolved":
        if latest_review.resolution == "approve":
            return session, "manual", latest_review
        if latest_review.resolution in {"reject", "request_retry"}:
            return None

    if session.status == "verified" and session.decision == "match":
        return session, "system", latest_review

    return None
