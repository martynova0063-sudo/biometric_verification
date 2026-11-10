"""Add authenticated sessions and employee photo upload grants."""
from alembic import op
import sqlalchemy as sa


revision = "0009_auth_employee_photos"
down_revision = "0008_data_drift"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "auth_sessions",
        sa.Column("sid_hash", sa.String(length=64), nullable=False),
        sa.Column("user_id", sa.String(length=255), nullable=False),
        sa.Column("email", sa.String(length=320), nullable=True),
        sa.Column("first_name", sa.String(length=160), nullable=True),
        sa.Column("last_name", sa.String(length=160), nullable=True),
        sa.Column("profile_image_url", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("sid_hash"),
    )
    op.create_index("ix_auth_sessions_user_id", "auth_sessions", ["user_id"])
    op.create_index("ix_auth_sessions_expires_at", "auth_sessions", ["expires_at"])

    op.create_table(
        "employee_photo_uploads",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("employee_guid", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.String(length=255), nullable=False),
        sa.Column("object_path", sa.Text(), nullable=False),
        sa.Column("content_type", sa.String(length=80), nullable=False),
        sa.Column("expected_size", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("object_path"),
    )
    op.create_index(
        "ix_employee_photo_uploads_employee_guid",
        "employee_photo_uploads",
        ["employee_guid"],
    )
    op.create_index(
        "ix_employee_photo_uploads_user_id",
        "employee_photo_uploads",
        ["user_id"],
    )
    op.create_index(
        "ix_employee_photo_uploads_expires_at",
        "employee_photo_uploads",
        ["expires_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_employee_photo_uploads_expires_at", table_name="employee_photo_uploads")
    op.drop_index("ix_employee_photo_uploads_user_id", table_name="employee_photo_uploads")
    op.drop_index("ix_employee_photo_uploads_employee_guid", table_name="employee_photo_uploads")
    op.drop_table("employee_photo_uploads")
    op.drop_index("ix_auth_sessions_expires_at", table_name="auth_sessions")
    op.drop_index("ix_auth_sessions_user_id", table_name="auth_sessions")
    op.drop_table("auth_sessions")
