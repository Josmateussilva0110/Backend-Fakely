from datetime import datetime, timezone

from app.models.evidence_model import Evidence
from app.models.news_analysis_model import Classification, NewsAnalysis, VerificationStatus
from app.models.news_analysis_repository import NewsAnalysisRepository
from app.models.news_model import News


def make_analysis(news: News, classification: Classification, evidences: int = 0) -> NewsAnalysis:
    return NewsAnalysis(
        news=news, classification=classification, justification="j", limitations=[], style_indicators=[],
        keywords=[], sources_checked=[], sources_failed=[], verification_status=VerificationStatus.COMPLETE,
        method="evidence_rules", model_version="v1",
        evidences=[
            Evidence(position=i, source_name="Lupa", organization="lupa", category="fact_checker",
                     retrieved_via="Lupa", title="t", url=f"https://lupa.org/{i}", excerpt="",
                     accessed_at=datetime.now(timezone.utc), relevance=0.9, stance="contradicts",
                     independent=True)
            for i in range(evidences, 0, -1)
        ],
    )


def test_save_and_find_with_ordered_evidences(db_session):
    repository = NewsAnalysisRepository(db_session)
    saved = repository.save(make_analysis(News(text="texto"), Classification.LIKELY_FALSE, evidences=2))
    db_session.expunge_all()
    found = repository.find_by_id(saved.id)
    assert [e.position for e in found.evidences] == [1, 2]
    assert found.news.text == "texto"


def test_find_page_filters_by_news_and_classification(db_session):
    repository = NewsAnalysisRepository(db_session)
    first, second = News(text="a"), News(text="b")
    repository.save(make_analysis(first, Classification.LIKELY_FALSE))
    repository.save(make_analysis(first, Classification.INCONCLUSIVE))
    repository.save(make_analysis(second, Classification.LIKELY_FALSE))

    items, total = repository.find_page(offset=0, limit=10, order_by="-analyzed_at", news_id=first.id)
    assert total == 2
    _, total = repository.find_page(
        offset=0, limit=10, order_by="analyzed_at", classification=Classification.LIKELY_FALSE
    )
    assert total == 2
