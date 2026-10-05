"""add verification to news

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-04
"""
import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("news", sa.Column("verification", sa.JSON(), nullable=True))
    # O método passou a combinar etapas (ex.: "model+heuristic+verification")
    op.alter_column("news", "method", type_=sa.String(50), existing_nullable=False)


def downgrade() -> None:
    op.alter_column("news", "method", type_=sa.String(30), existing_nullable=False)
    op.drop_column("news", "verification")
