"""store the sportsbook home-win probability and the model's own winner pick

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-28 09:00:00
"""

from alembic import op
import sqlalchemy as sa


revision = '0003'
down_revision = '0002'
branch_labels = None
depends_on = None

_PREDICTION_COLUMNS = (
    ("winner_source", sa.String(length=10)),
    ("model_winner_pick", sa.String(length=10)),
    ("model_winner_trust", sa.Float()),
)


def upgrade() -> None:
    with op.batch_alter_table("games") as batch_op:
        batch_op.add_column(sa.Column("home_win_prob", sa.Float(), nullable=True))
    with op.batch_alter_table("predictions") as batch_op:
        for name, type_ in _PREDICTION_COLUMNS:
            batch_op.add_column(sa.Column(name, type_, nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("predictions") as batch_op:
        for name, _ in _PREDICTION_COLUMNS:
            batch_op.drop_column(name)
    with op.batch_alter_table("games") as batch_op:
        batch_op.drop_column("home_win_prob")
