import asyncio

from app.core import container
from app.core.container import AppContainer


class FakeEmbeddingService:
    pass


class FakeVectorStore:
    def __init__(self, embedding_backend):
        self.embedding_backend = embedding_backend
        self.persisted = False

    async def persist(self):
        self.persisted = True


def test_container_defers_embedding_and_vector_store_creation(monkeypatch):
    monkeypatch.setattr(container, "LocalEmbeddingBackend", FakeEmbeddingService)
    monkeypatch.setattr(container, "VectorStoreService", FakeVectorStore)
    monkeypatch.setattr(container.settings, "EMBEDDING_PROVIDER", "local")

    app_container = AppContainer(llm_service=object())

    assert app_container.embedding_backend is None
    assert app_container.vector_store is None

    embedding_backend = app_container.get_embedding_backend()
    vector_store = app_container.get_vector_store()

    assert embedding_backend is app_container.get_embedding_backend()
    assert vector_store is app_container.get_vector_store()
    assert vector_store.embedding_backend is embedding_backend


def test_container_selects_gemini_embedding_backend(monkeypatch):
    monkeypatch.setattr(container.settings, "EMBEDDING_PROVIDER", "gemini")

    app_container = AppContainer(llm_service=object())

    assert isinstance(
        app_container.get_embedding_backend(), container.GeminiEmbeddingBackend
    )


def test_container_persists_vector_store_only_after_it_is_loaded(monkeypatch):
    monkeypatch.setattr(container, "LocalEmbeddingBackend", FakeEmbeddingService)
    monkeypatch.setattr(container, "VectorStoreService", FakeVectorStore)
    monkeypatch.setattr(container.settings, "EMBEDDING_PROVIDER", "local")

    app_container = AppContainer(llm_service=object())
    asyncio.run(app_container.persist_vector_store_if_loaded())

    assert app_container.vector_store is None

    vector_store = app_container.get_vector_store()
    asyncio.run(app_container.persist_vector_store_if_loaded())

    assert vector_store.persisted is True
