"""Allow unavailable biometric evaluations to store unknown scores as NULL."""
from alembic import op
import sqlalchemy as sa


revision = "0007_not_evaluable"
down_revision = "0006_employee_directory"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("verification_sessions") as batch_op:
        batch_op.alter_column(
            "match_score",
            existing_type=sa.Float(),
            nullable=True,
        )
        batch_op.alter_column(
            "liveness_score",
            existing_type=sa.Float(),
            nullable=True,
        )
        batch_op.alter_column(
            "quality_score",
            existing_type=sa.Float(),
            nullable=True,
        )


def downgrade() -> None:
    connection = op.get_bind()
    nullable_scores = connection.execute(sa.text(
        "SELECT 1 FROM verification_sessions "
        "WHERE match_score IS NULL OR liveness_score IS NULL OR quality_score IS NULL "
        "LIMIT 1"
    )).first()
    if nullable_scores:
        raise RuntimeError(
            "Cannot downgrade while a verification session has an unknown score."
        )

    with op.batch_alter_table("verification_sessions") as batch_op:
        batch_op.alter_column(
            "match_score",
            existing_type=sa.Float(),
            nullable=False,
        )
        batch_op.alter_column(
            "liveness_score",
            existing_type=sa.Float(),
            nullable=False,
        )
        batch_op.alter_column(
            "quality_score",
            existing_type=sa.Float(),
            nullable=False,
        )
