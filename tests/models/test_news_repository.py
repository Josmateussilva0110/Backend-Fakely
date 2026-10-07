from datetime import date

from app.models.news_model import News
from app.models.news_repository import NewsRepository


def test_find_page_filters_orders_and_paginates(db_session):
    repository = NewsRepository(db_session)
    for day in (1, 3, 2):
        repository.save(News(text="texto", source="Blog.net", published_at=date(2026, 9, day)))
    repository.save(News(text="texto", source="outro.com"))

    items, total = repository.find_page(offset=0, limit=2, order_by="-published_at", source="blog.net")
    assert total == 3
    assert [n.published_at.day for n in items] == [3, 2]
