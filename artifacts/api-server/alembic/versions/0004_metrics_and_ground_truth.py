"""Persist automatic biometric outcomes, operator labels, and metric snapshots."""
from alembic import op
import sqlalchemy as sa


revision = "0004_metrics_and_ground_truth"
down_revision = "0003_registered_models"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("verification_sessions", sa.Column("is_match", sa.Boolean(), nullable=True))
    op.add_column("verification_sessions", sa.Column("ground_truth_match", sa.Boolean(), nullable=True))
    op.add_column("verification_sessions", sa.Column("is_live", sa.Boolean(), nullable=True))
    op.add_column("verification_sessions", sa.Column("ground_truth_liveness", sa.Boolean(), nullable=True))
    op.add_column("verification_sessions", sa.Column("cosine_distance", sa.Float(), nullable=True))
    op.add_column("review_tasks", sa.Column("verification_session_id", sa.String(length=36), nullable=True))
    op.create_index(
        "ix_review_tasks_verification_session_id",
        "review_tasks",
        ["verification_session_id"],
    )

    bind = op.get_bind()
    sessions = sa.Table("verification_sessions", sa.MetaData(), autoload_with=bind)
    for row in bind.execute(sa.select(sessions.c.id, sessions.c.match_score, sessions.c.decision, sessions.c.details)):
        details = row.details or {}
        face_match = details.get("face_match") or {}
        liveness = details.get("liveness") or {}
        raw_score = face_match.get("match_score", row.match_score)
        threshold = face_match.get("threshold")
        is_match = (
            float(raw_score) >= float(threshold)
            if raw_score is not None and threshold is not None
            else row.decision == "match"
        )
        liveness_decision = liveness.get("decision")
        is_live = liveness_decision == "live" if liveness_decision is not None else None
        bind.execute(
            sessions.update()
            .where(sessions.c.id == row.id)
            .values(is_match=is_match, is_live=is_live, cosine_distance=raw_score)
        )

    review_tasks = sa.Table("review_tasks", sa.MetaData(), autoload_with=bind)
    session_ids_by_employee: dict[str, list[str]] = {}
    for row in bind.execute(sa.select(sessions.c.id, sessions.c.employee_ref)):
        session_ids_by_employee.setdefault(row.employee_ref, []).append(row.id)
    for task in bind.execute(
        sa.select(review_tasks.c.id, review_tasks.c.subject_ref)
        .where(review_tasks.c.kind == "verification", review_tasks.c.verification_session_id.is_(None))
    ):
        matching_ids = session_ids_by_employee.get(task.subject_ref, [])
        if len(matching_ids) == 1:
            bind.execute(
                review_tasks.update()
                .where(review_tasks.c.id == task.id)
                .values(verification_session_id=matching_ids[0])
            )

    op.create_table(
        "biometric_metrics_log",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("calculation_date", sa.Date(), nullable=False),
        sa.Column("window_start", sa.DateTime(), nullable=False),
        sa.Column("window_end", sa.DateTime(), nullable=False),
        sa.Column("far", sa.Numeric(5, 4), nullable=False),
        sa.Column("frr", sa.Numeric(5, 4), nullable=False),
        sa.Column("cer", sa.Numeric(5, 4), nullable=False),
        sa.Column("apcer", sa.Numeric(5, 4), nullable=False),
        sa.Column("bpcer", sa.Numeric(5, 4), nullable=False),
        sa.Column("acer", sa.Numeric(5, 4), nullable=False),
        sa.Column("total_samples", sa.Integer(), nullable=False),
        sa.Column("successful_matches", sa.Integer(), nullable=False),
        sa.Column("failed_matches", sa.Integer(), nullable=False),
        sa.Column("spoof_attempts_detected", sa.Integer(), nullable=False),
        sa.Column("live_rejected_as_spoof", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.current_timestamp(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("biometric_metrics_log")
    op.drop_index("ix_review_tasks_verification_session_id", table_name="review_tasks")
    op.drop_column("review_tasks", "verification_session_id")
    op.drop_column("verification_sessions", "cosine_distance")
    op.drop_column("verification_sessions", "ground_truth_liveness")
    op.drop_column("verification_sessions", "is_live")
    op.drop_column("verification_sessions", "ground_truth_match")
    op.drop_column("verification_sessions", "is_match")
