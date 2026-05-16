import asyncio

from app.core import container
from app.core.container import AppContainer


class FakeEmbeddingService:
    pass


class FakeVectorStore:
    def __init__(self, embedding_service):
        self.embedding_service = embedding_service
        self.persisted = False

    async def persist(self):
        self.persisted = True


def test_container_defers_embedding_and_vector_store_creation(monkeypatch):
    monkeypatch.setattr(container, "EmbeddingService", FakeEmbeddingService)
    monkeypatch.setattr(container, "VectorStoreService", FakeVectorStore)

    app_container = AppContainer(llm_service=object())

    assert app_container.embedding_service is None
    assert app_container.vector_store is None

    embedding_service = app_container.get_embedding_service()
    vector_store = app_container.get_vector_store()

    assert embedding_service is app_container.get_embedding_service()
    assert vector_store is app_container.get_vector_store()
    assert vector_store.embedding_service is embedding_service


def test_container_persists_vector_store_only_after_it_is_loaded(monkeypatch):
    monkeypatch.setattr(container, "EmbeddingService", FakeEmbeddingService)
    monkeypatch.setattr(container, "VectorStoreService", FakeVectorStore)

    app_container = AppContainer(llm_service=object())
    asyncio.run(app_container.persist_vector_store_if_loaded())

    assert app_container.vector_store is None

    vector_store = app_container.get_vector_store()
    asyncio.run(app_container.persist_vector_store_if_loaded())

    assert vector_store.persisted is True
