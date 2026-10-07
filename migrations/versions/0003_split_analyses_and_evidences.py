"""split analyses and evidences from news

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-06
"""
import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None

LEGACY_JUSTIFICATION = (
    "Análise migrada da versão anterior, baseada principalmente no estilo de escrita. "
    "Solicite uma nova análise para obter o resultado baseado em evidências."
)


def upgrade() -> None:
    op.create_table(
        "news_analyses",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("news_id", sa.Integer(), sa.ForeignKey("news.id", ondelete="CASCADE"), nullable=False),
        sa.Column("claim", sa.Text()),
        sa.Column("classification", sa.String(20), nullable=False),
        sa.Column("justification", sa.Text(), nullable=False),
        sa.Column("limitations", sa.JSON(), nullable=False),
        sa.Column("style_indicators", sa.JSON(), nullable=False),
        sa.Column("keywords", sa.JSON(), nullable=False),
        sa.Column("sources_checked", sa.JSON(), nullable=False),
        sa.Column("sources_failed", sa.JSON(), nullable=False),
        sa.Column("verification_status", sa.String(20), nullable=False),
        sa.Column("source_assessment", sa.JSON()),
        sa.Column("method", sa.String(50), nullable=False),
        sa.Column("model_version", sa.String(50), nullable=False),
        sa.Column("analyzed_at", sa.DateTime(timezone=True), nullable=False),
    )
    for column in ("news_id", "classification", "verification_status", "analyzed_at"):
        op.create_index(f"ix_news_analyses_{column}", "news_analyses", [column])

    op.create_table(
        "evidences",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "analysis_id", sa.Integer(), sa.ForeignKey("news_analyses.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("source_name", sa.String(255), nullable=False),
        sa.Column("organization", sa.String(255), nullable=False),
        sa.Column("category", sa.String(20), nullable=False),
        sa.Column("retrieved_via", sa.String(255), nullable=False),
        sa.Column("title", sa.String(1000), nullable=False),
        sa.Column("url", sa.String(2048), nullable=False),
        sa.Column("excerpt", sa.Text(), nullable=False),
        sa.Column("published_at", sa.Date()),
        sa.Column("accessed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("relevance", sa.Float(), nullable=False),
        sa.Column("stance", sa.String(20), nullable=False),
        sa.Column("verdict", sa.String(10)),
        sa.Column("rating", sa.String(255)),
        sa.Column("independent", sa.Boolean(), nullable=False),
    )
    op.create_index("ix_evidences_analysis_id", "evidences", ["analysis_id"])

    # Preserva o histórico: cada notícia antiga vira uma análise "legada"
    op.execute(
        sa.text(
            "INSERT INTO news_analyses (news_id, classification, justification, limitations, "
            "style_indicators, keywords, sources_checked, sources_failed, verification_status, "
            "method, model_version, analyzed_at) "
            "SELECT id, classification, :justification, '[]', evidence, '[]', '[]', '[]', 'skipped', "
            "method, 'legacy', analyzed_at FROM news"
        ).bindparams(justification=LEGACY_JUSTIFICATION)
    )

    op.alter_column("news", "analyzed_at", new_column_name="created_at")
    op.drop_index("ix_news_analyzed_at", table_name="news")
    op.create_index("ix_news_created_at", "news", ["created_at"])
    op.drop_index("ix_news_classification", table_name="news")
    op.drop_index("ix_news_confidence", table_name="news")
    for column in ("classification", "confidence", "probability_fake", "method", "evidence", "features", "verification"):
        op.drop_column("news", column)


def downgrade() -> None:
    op.add_column("news", sa.Column("verification", sa.JSON(), nullable=True))
    op.add_column("news", sa.Column("features", sa.JSON(), nullable=False, server_default="{}"))
    op.add_column("news", sa.Column("evidence", sa.JSON(), nullable=False, server_default="[]"))
    op.add_column("news", sa.Column("method", sa.String(50), nullable=False, server_default="legacy"))
    op.add_column("news", sa.Column("probability_fake", sa.Float()))
    op.add_column("news", sa.Column("confidence", sa.Float()))
    op.add_column(
        "news", sa.Column("classification", sa.String(20), nullable=False, server_default="inconclusive")
    )
    # Restaura a classificação da análise mais recente de cada notícia
    op.execute(
        "UPDATE news SET classification = (SELECT a.classification FROM news_analyses a "
        "WHERE a.news_id = news.id ORDER BY a.analyzed_at DESC LIMIT 1) "
        "WHERE EXISTS (SELECT 1 FROM news_analyses a WHERE a.news_id = news.id)"
    )
    op.create_index("ix_news_classification", "news", ["classification"])
    op.create_index("ix_news_confidence", "news", ["confidence"])
    op.drop_index("ix_news_created_at", table_name="news")
    op.alter_column("news", "created_at", new_column_name="analyzed_at")
    op.create_index("ix_news_analyzed_at", "news", ["analyzed_at"])
    op.drop_table("evidences")
    op.drop_table("news_analyses")
