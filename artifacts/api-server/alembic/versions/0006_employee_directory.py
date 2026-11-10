"""Add employee directory and reference photo metadata."""
from alembic import op
import sqlalchemy as sa


revision = "0006_employee_directory"
down_revision = "0005_metric_calculation_details"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "employees",
        sa.Column("guid", sa.String(length=36), primary_key=True),
        sa.Column("personnel_number", sa.String(length=40), nullable=False),
        sa.Column("sex", sa.String(length=20), nullable=False),
        sa.Column("surname", sa.String(length=120), nullable=False),
        sa.Column("given_name", sa.String(length=120), nullable=False),
        sa.Column("patronymic", sa.String(length=120), nullable=False),
        sa.Column("birth_date", sa.String(length=80), nullable=False),
        sa.Column("organization", sa.String(length=200), nullable=False),
        sa.Column("is_photo", sa.Boolean(), nullable=False),
        sa.Column("is_test", sa.Boolean(), nullable=False),
        sa.Column("photo_path", sa.Text(), nullable=True),
    )
    op.create_index("ix_employees_personnel_number", "employees", ["personnel_number"], unique=True)
    op.create_index("ix_employees_surname", "employees", ["surname"])
    op.create_index("ix_employees_organization", "employees", ["organization"])
    op.create_index("ix_employees_is_test", "employees", ["is_test"])


def downgrade() -> None:
    op.drop_index("ix_employees_is_test", table_name="employees")
    op.drop_index("ix_employees_organization", table_name="employees")
    op.drop_index("ix_employees_surname", table_name="employees")
    op.drop_index("ix_employees_personnel_number", table_name="employees")
    op.drop_table("employees")
