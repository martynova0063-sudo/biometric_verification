"""Persist editable model versions and biometric configuration."""
from alembic import op
import sqlalchemy as sa


revision = "0003_registered_models"
down_revision = "0002_verification_details"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "registered_models",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("model_name", sa.String(length=120), nullable=False),
        sa.Column("version", sa.String(length=60), nullable=False, unique=True),
        sa.Column("status", sa.String(length=30), nullable=False, server_default="draft"),
        sa.Column("purpose", sa.Text(), nullable=False),
        sa.Column("change_summary", sa.Text(), nullable=False),
        sa.Column("parameters", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_registered_models_status", "registered_models", ["status"])


def downgrade() -> None:
    op.drop_index("ix_registered_models_status", table_name="registered_models")
    op.drop_table("registered_models")