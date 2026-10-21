"""initial schema (as created by the original create_all)

Revision ID: 0001
Revises: 
Create Date: 2026-10-21 09:23:14.089121
"""

from alembic import op
import sqlalchemy as sa


revision = '0001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('teams',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('api_sports_id', sa.Integer(), nullable=False, comment='Unique team identifier assigned by API-Sports.'),
    sa.Column('name', sa.String(length=120), nullable=False, comment="Full franchise name, e.g. 'Los Angeles Lakers'."),
    sa.Column('abbreviation', sa.String(length=10), nullable=True, comment="Short ticker-style code, e.g. 'LAL'."),
    sa.Column('conference', sa.String(length=20), nullable=True, comment="'East' or 'West'."),
    sa.Column('division', sa.String(length=40), nullable=True, comment="Division name, e.g. 'Pacific'."),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('teams', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_teams_api_sports_id'), ['api_sports_id'], unique=True)

    op.create_table('games',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('api_sports_id', sa.Integer(), nullable=False, comment='Unique game identifier assigned by API-Sports.'),
    sa.Column('date', sa.DateTime(timezone=True), nullable=False, comment='Scheduled tip-off time (UTC).'),
    sa.Column('home_team_id', sa.Integer(), nullable=False),
    sa.Column('away_team_id', sa.Integer(), nullable=False),
    sa.Column('status', sa.String(length=20), nullable=False, comment="One of: 'scheduled', 'live', 'finished'."),
    sa.Column('season', sa.String(length=20), nullable=False, comment="Season string, e.g. '2025-2026'."),
    sa.Column('home_score', sa.Integer(), nullable=True, comment='Final home team score (populated after game ends).'),
    sa.Column('away_score', sa.Integer(), nullable=True, comment='Final away team score (populated after game ends).'),
    sa.Column('home_halftime_score', sa.Integer(), nullable=True, comment='Home team score at the end of the first half.'),
    sa.Column('away_halftime_score', sa.Integer(), nullable=True, comment='Away team score at the end of the first half.'),
    sa.Column('ou_line', sa.Float(), nullable=True, comment='Sportsbook full-game over/under points total.'),
    sa.Column('ht_ou_line', sa.Float(), nullable=True, comment='Sportsbook first-half over/under points total.'),
    sa.ForeignKeyConstraint(['away_team_id'], ['teams.id'], ondelete='RESTRICT'),
    sa.ForeignKeyConstraint(['home_team_id'], ['teams.id'], ondelete='RESTRICT'),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('games', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_games_api_sports_id'), ['api_sports_id'], unique=True)
        batch_op.create_index(batch_op.f('ix_games_away_team_id'), ['away_team_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_games_home_team_id'), ['home_team_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_games_season'), ['season'], unique=False)

    op.create_table('team_stats',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('team_id', sa.Integer(), nullable=False),
    sa.Column('season', sa.String(length=20), nullable=False, comment="Season string matching Game.season, e.g. '2025-2026'."),
    sa.Column('games_played', sa.Integer(), nullable=False),
    sa.Column('wins', sa.Integer(), nullable=False),
    sa.Column('losses', sa.Integer(), nullable=False),
    sa.Column('ppg', sa.Float(), nullable=False, comment='Points per game (offensive average).'),
    sa.Column('oppg', sa.Float(), nullable=False, comment='Opponent points per game allowed (defensive average).'),
    sa.Column('home_wins', sa.Integer(), nullable=False),
    sa.Column('home_losses', sa.Integer(), nullable=False),
    sa.Column('away_wins', sa.Integer(), nullable=False),
    sa.Column('away_losses', sa.Integer(), nullable=False),
    sa.Column('fg_pct', sa.Float(), nullable=True, comment='Field-goal percentage (0–100 scale).'),
    sa.Column('three_pct', sa.Float(), nullable=True, comment='Three-point percentage (0–100 scale).'),
    sa.Column('ft_pct', sa.Float(), nullable=True, comment='Free-throw percentage (0–100 scale).'),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, comment='Timestamp of the last ingestion update (UTC).'),
    sa.ForeignKeyConstraint(['team_id'], ['teams.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('team_id', 'season', name='uq_team_stats_team_season')
    )
    with op.batch_alter_table('team_stats', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_team_stats_team_id'), ['team_id'], unique=False)

    op.create_table('predictions',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('game_id', sa.Integer(), nullable=False, comment='One-to-one link back to the game being predicted.'),
    sa.Column('winner_pick', sa.String(length=10), nullable=False, comment="Predicted winner side: 'Home' or 'Away'."),
    sa.Column('winner_trust', sa.Float(), nullable=False, comment='Model confidence for winner pick, in [0, 100].'),
    sa.Column('ft_ou_pick', sa.String(length=10), nullable=False, comment="Full-game O/U pick: 'Over' or 'Under'."),
    sa.Column('ft_ou_trust', sa.Float(), nullable=False, comment='Model confidence for the full-game O/U pick, in [0, 100].'),
    sa.Column('ht_ou_pick', sa.String(length=10), nullable=False, comment="First-half O/U pick: 'Over' or 'Under'."),
    sa.Column('ht_ou_trust', sa.Float(), nullable=False, comment='Model confidence for the first-half O/U pick, in [0, 100].'),
    sa.Column('model_version', sa.String(length=40), nullable=False, comment='Identifier for the model/algorithm version that created this row.'),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, comment='Timestamp when the prediction was generated (UTC).'),
    sa.ForeignKeyConstraint(['game_id'], ['games.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('predictions', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_predictions_game_id'), ['game_id'], unique=True)



def downgrade() -> None:
    with op.batch_alter_table('predictions', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_predictions_game_id'))

    op.drop_table('predictions')
    with op.batch_alter_table('team_stats', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_team_stats_team_id'))

    op.drop_table('team_stats')
    with op.batch_alter_table('games', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_games_season'))
        batch_op.drop_index(batch_op.f('ix_games_home_team_id'))
        batch_op.drop_index(batch_op.f('ix_games_away_team_id'))
        batch_op.drop_index(batch_op.f('ix_games_api_sports_id'))

    op.drop_table('games')
    with op.batch_alter_table('teams', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_teams_api_sports_id'))

    op.drop_table('teams')
