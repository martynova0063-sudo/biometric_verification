"""Persist detailed biometric verification results."""
from alembic import op
import sqlalchemy as sa

revision = "0002_verification_details"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "verification_sessions",
        sa.Column("details", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("verification_sessions", "details")