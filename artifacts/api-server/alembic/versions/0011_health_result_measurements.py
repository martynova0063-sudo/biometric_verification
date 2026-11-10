"""Store source measurements for employee-specific baselines."""
from alembic import op
import sqlalchemy as sa


revision = "0011_health_result_measurements"
down_revision = "0010_employee_photo_archives"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "health_results",
        sa.Column("measurements", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("health_results", "measurements")
