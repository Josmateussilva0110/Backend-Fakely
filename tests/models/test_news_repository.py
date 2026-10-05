from app.models.news_model import Classification, News
from app.models.news_repository import NewsRepository


def make_news(classification: Classification, confidence: float, source: str | None = None) -> News:
    return News(text="texto", classification=classification, confidence=confidence,
                method="heuristic", source=source)


def test_find_page_filters_orders_and_paginates(db_session):
    repository = NewsRepository(db_session)
    for confidence in (0.7, 0.9, 0.8):
        repository.save(make_news(Classification.LIKELY_FALSE, confidence, source="Blog.net"))
    repository.save(make_news(Classification.LIKELY_TRUE, 0.95))

    items, total = repository.find_page(
        offset=0, limit=2, order_by="-confidence", classification=Classification.LIKELY_FALSE,
        source="blog.net",
    )
    assert total == 3
    assert [n.confidence for n in items] == [0.9, 0.8]
