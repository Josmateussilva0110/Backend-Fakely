"""add entailment to evidences

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-16
"""
import sqlalchemy as sa
from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Probabilidade de a manchete implicar a afirmação (modelo NLI); vazio nas análises antigas
    op.add_column("evidences", sa.Column("entailment", sa.Float()))


def downgrade() -> None:
    op.drop_column("evidences", "entailment")
