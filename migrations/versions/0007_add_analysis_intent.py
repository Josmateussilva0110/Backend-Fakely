"""add intent to news_analyses

Revision ID: 0007
Revises: 0006
Create Date: 2026-11-22
"""
import sqlalchemy as sa
from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Intenção do texto (claim, open_question, prediction...); vazio nas análises antigas
    op.add_column("news_analyses", sa.Column("intent", sa.String(20)))


def downgrade() -> None:
    op.drop_column("news_analyses", "intent")
