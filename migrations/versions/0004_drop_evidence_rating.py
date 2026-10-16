"""drop rating from evidences

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-16
"""
import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # A classificação textual vinha só do Google Fact Check, removido do projeto
    op.drop_column("evidences", "rating")


def downgrade() -> None:
    op.add_column("evidences", sa.Column("rating", sa.String(255)))
