"""Add persistent data drift observations, baselines and daily runs."""
from alembic import op
import sqlalchemy as sa


revision = "0008_data_drift"
down_revision = "0007_not_evaluable"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "data_drift_observations",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("source_type", sa.String(length=30), nullable=False),
        sa.Column("observed_at", sa.DateTime(), nullable=False),
        sa.Column("is_test", sa.Boolean(), nullable=False),
        sa.Column("metrics", sa.JSON(), nullable=False),
        sa.Column("capture_metadata", sa.JSON(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_data_drift_observations_observed_at", "data_drift_observations", ["observed_at"])
    op.create_index("ix_data_drift_observations_is_test", "data_drift_observations", ["is_test"])

    op.create_table(
        "data_drift_baselines",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("window_start", sa.DateTime(), nullable=False),
        sa.Column("window_end", sa.DateTime(), nullable=False),
        sa.Column("distributions", sa.JSON(), nullable=False),
        sa.Column("observation_count", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "data_drift_runs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("calculation_date", sa.Date(), nullable=False),
        sa.Column("window_start", sa.DateTime(), nullable=False),
        sa.Column("window_end", sa.DateTime(), nullable=False),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("psi_values", sa.JSON(), nullable=False),
        sa.Column("sample_counts", sa.JSON(), nullable=False),
        sa.Column("needs_recalibration", sa.Boolean(), nullable=False),
        sa.Column("manual_review_mode", sa.Boolean(), nullable=False),
        sa.Column("alert_messages", sa.JSON(), nullable=False),
        sa.Column("recalibration", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("calculation_date"),
    )
    op.create_index("ix_data_drift_runs_calculation_date", "data_drift_runs", ["calculation_date"])

    op.create_table(
        "data_drift_control",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("manual_review_mode", sa.Boolean(), nullable=False),
        sa.Column("manual_review_reason", sa.Text(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("data_drift_control")
    op.drop_index("ix_data_drift_runs_calculation_date", table_name="data_drift_runs")
    op.drop_table("data_drift_runs")
    op.drop_table("data_drift_baselines")
    op.drop_index("ix_data_drift_observations_is_test", table_name="data_drift_observations")
    op.drop_index("ix_data_drift_observations_observed_at", table_name="data_drift_observations")
    op.drop_table("data_drift_observations")
