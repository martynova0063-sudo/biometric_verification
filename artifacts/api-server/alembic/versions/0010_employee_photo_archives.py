"""Keep replaced employee photos in the private training dataset."""
from alembic import op
import sqlalchemy as sa


revision = "0010_employee_photo_archives"
down_revision = "0009_auth_employee_photos"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "employee_photo_archives",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("employee_guid", sa.String(length=36), nullable=False),
        sa.Column("source_photo_path", sa.Text(), nullable=False),
        sa.Column("object_path", sa.Text(), nullable=False),
        sa.Column("archived_by", sa.String(length=255), nullable=False),
        sa.Column("content_type", sa.String(length=80), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["employee_guid"], ["employees.guid"], ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("object_path"),
    )
    op.create_index(
        "ix_employee_photo_archives_employee_guid",
        "employee_photo_archives",
        ["employee_guid"],
    )
    op.create_index(
        "ix_employee_photo_archives_archived_by",
        "employee_photo_archives",
        ["archived_by"],
    )
    op.create_index(
        "ix_employee_photo_archives_created_at",
        "employee_photo_archives",
        ["created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_employee_photo_archives_created_at", table_name="employee_photo_archives")
    op.drop_index("ix_employee_photo_archives_archived_by", table_name="employee_photo_archives")
    op.drop_index("ix_employee_photo_archives_employee_guid", table_name="employee_photo_archives")
    op.drop_table("employee_photo_archives")
