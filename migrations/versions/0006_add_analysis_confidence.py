"""add confidence to news_analyses

Revision ID: 0006
Revises: 0005
Create Date: 2026-10-22
"""
import sqlalchemy as sa
from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Nível de confiança (high/medium/low); vazio nas análises antigas
    op.add_column("news_analyses", sa.Column("confidence", sa.String(10)))


def downgrade() -> None:
    op.drop_column("news_analyses", "confidence")
