"""allow NULL O/U picks (no line available / model not validated)

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-21 09:30:00
"""

from alembic import op
import sqlalchemy as sa


revision = '0002'
down_revision = '0001'
branch_labels = None
depends_on = None

_COLUMNS = (
    ("ft_ou_pick", sa.String(length=10)),
    ("ft_ou_trust", sa.Float()),
    ("ht_ou_pick", sa.String(length=10)),
    ("ht_ou_trust", sa.Float()),
)


def upgrade() -> None:
    with op.batch_alter_table("predictions") as batch_op:
        for name, type_ in _COLUMNS:
            batch_op.alter_column(name, existing_type=type_, nullable=True)


def downgrade() -> None:
    # Rows without an O/U pick cannot satisfy NOT NULL; drop them first.
    op.execute("DELETE FROM predictions WHERE ft_ou_pick IS NULL OR ht_ou_pick IS NULL")
    with op.batch_alter_table("predictions") as batch_op:
        for name, type_ in _COLUMNS:
            batch_op.alter_column(name, existing_type=type_, nullable=False)
