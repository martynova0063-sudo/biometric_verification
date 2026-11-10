"""Initial IS UKZ schema."""
from alembic import op
import sqlalchemy as sa

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "health_results",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("employee_ref", sa.String(120), nullable=False),
        sa.Column("event_date", sa.DateTime(), nullable=False),
        sa.Column("health_id", sa.Float(), nullable=False),
        sa.Column("category", sa.String(30), nullable=False),
        sa.Column("status", sa.String(30), nullable=False),
        sa.Column("model_version", sa.String(40), nullable=False),
        sa.Column("components", sa.JSON(), nullable=False),
        sa.Column("evidence", sa.JSON(), nullable=False),
        sa.Column("disclaimer", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_health_results_employee_ref", "health_results", ["employee_ref"])
    op.create_index("ix_health_results_event_date", "health_results", ["event_date"])
    op.create_index("ix_health_results_category", "health_results", ["category"])
    op.create_index("ix_health_results_created_at", "health_results", ["created_at"])
    op.create_table(
        "verification_sessions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("employee_ref", sa.String(120), nullable=False),
        sa.Column("status", sa.String(30), nullable=False),
        sa.Column("match_score", sa.Float(), nullable=False),
        sa.Column("liveness_score", sa.Float(), nullable=False),
        sa.Column("quality_score", sa.Float(), nullable=False),
        sa.Column("decision", sa.String(30), nullable=False),
        sa.Column("algorithm_version", sa.String(40), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("audit_ref", sa.String(80), nullable=False),
        sa.Column("attack_type", sa.String(80), nullable=True),
    )
    op.create_index("ix_verification_sessions_employee_ref", "verification_sessions", ["employee_ref"])
    op.create_index("ix_verification_sessions_status", "verification_sessions", ["status"])
    op.create_index("ix_verification_sessions_created_at", "verification_sessions", ["created_at"])
    op.create_table(
        "review_tasks",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("kind", sa.String(30), nullable=False),
        sa.Column("subject_ref", sa.String(120), nullable=False),
        sa.Column("status", sa.String(30), nullable=False),
        sa.Column("priority", sa.String(20), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("resolved_at", sa.DateTime(), nullable=True),
        sa.Column("resolution", sa.String(40), nullable=True),
        sa.Column("reviewer_note", sa.Text(), nullable=True),
    )
    op.create_index("ix_review_tasks_kind", "review_tasks", ["kind"])
    op.create_index("ix_review_tasks_subject_ref", "review_tasks", ["subject_ref"])
    op.create_index("ix_review_tasks_status", "review_tasks", ["status"])
    op.create_index("ix_review_tasks_created_at", "review_tasks", ["created_at"])
    op.create_table(
        "activity_items",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("type", sa.String(30), nullable=False),
        sa.Column("title", sa.String(180), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("status", sa.String(30), nullable=False),
    )
    op.create_index("ix_activity_items_created_at", "activity_items", ["created_at"])


def downgrade() -> None:
    op.drop_table("activity_items")
    op.drop_table("review_tasks")
    op.drop_table("verification_sessions")
    op.drop_table("health_results")