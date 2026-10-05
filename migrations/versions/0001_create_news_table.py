"""create news table

Revision ID: 0001
Revises:
Create Date: 2026-10-04
"""
import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "news",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("title", sa.String(500)),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("url", sa.String(2048)),
        sa.Column("source", sa.String(255)),
        sa.Column("published_at", sa.Date()),
        sa.Column("analyzed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("classification", sa.String(20), nullable=False),
        sa.Column("confidence", sa.Float()),
        sa.Column("probability_fake", sa.Float()),
        sa.Column("method", sa.String(30), nullable=False),
        sa.Column("evidence", sa.JSON(), nullable=False),
        sa.Column("features", sa.JSON(), nullable=False),
    )
    for column in ("source", "published_at", "analyzed_at", "classification", "confidence"):
        op.create_index(f"ix_news_{column}", "news", [column])


def downgrade() -> None:
    op.drop_table("news")
