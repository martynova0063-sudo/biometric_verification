"""Persist numerator and denominator counts for metric snapshots."""
from alembic import op
import sqlalchemy as sa


revision = "0005_metric_calculation_details"
down_revision = "0004_metrics_and_ground_truth"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "biometric_metrics_log",
        sa.Column("calculation_details", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("biometric_metrics_log", "calculation_details")
