import asyncio

import numpy as np

from app.services.embedding_service import EmbeddingService
from app.services import vector_store
from app.services.vector_store import VectorStoreService


class FakeModel:
    def encode(self, texts: list[str]):
        return [
            [1.0, 0.0, 0.0, 0.0] if text == "First text" else [0.0, 1.0, 0.0, 0.0]
            for text in texts
        ]


class FakeEmbeddingService:
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def embed(self, text: str):
        return self.embeddings[text]

    def embed_batch(self, texts: list[str]):
        return np.concatenate([self.embeddings[text] for text in texts], axis=0)


class FakePaper:
    def __init__(self, external_id: int, abstract: str):
        self.external_id = external_id
        self.abstract = abstract


class FakePaperService:
    def __init__(self, papers):
        self.papers = papers

    async def get_all_external_ids(self):
        return {paper.external_id for paper in self.papers}

    async def get_all_papers(self):
        return self.papers

    async def get_paper_by_external_id(self, external_id):
        for paper in self.papers:
            if paper.external_id == external_id:
                return paper
        return None


async def _exercise_vector_store(tmp_path, monkeypatch):
    index_path = tmp_path / "vector_index.faiss"
    monkeypatch.setattr(vector_store, "FAISS_INDEX_PATH", str(index_path))

    embeddings = {
        "First text": np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        "Second text": np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32),
        "Query": np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
    }
    monkeypatch.setattr(vector_store.settings, "EMBEDDING_DIM", 4)

    store = VectorStoreService(FakeEmbeddingService(embeddings))
    empty_contents = store.get_contents()
    empty_sync = await store.check_sync({1})

    await store.add_vector(1, "First text")
    await store.add_vector(1, "First text")
    await store.add_vector(2, "Second text")
    distances, indices = await store.search("Query", top_k=2)
    contents = store.get_contents()
    sync = await store.check_sync({1, 3})
    await store.persist()

    return {
        "empty_contents": empty_contents,
        "empty_sync": empty_sync,
        "length": len(store),
        "distances": distances,
        "indices": indices,
        "contents": contents,
        "sync": sync,
        "index_exists": index_path.exists(),
    }


def test_vector_store_add_search_contents_and_sync(tmp_path, monkeypatch):
    result = asyncio.run(_exercise_vector_store(tmp_path, monkeypatch))

    assert result["empty_contents"] == []
    assert result["empty_sync"] == {
        "is_sync": False,
        "missing_from_faiss": [1],
        "extra_in_faiss": [],
    }
    assert result["length"] == 2
    assert list(result["indices"]) == [1, 2]
    assert list(result["distances"]) == [1.0, 0.0]
    assert result["contents"] == [1, 2]
    assert result["sync"] == {
        "is_sync": False,
        "missing_from_faiss": [3],
        "extra_in_faiss": [2],
    }
    assert result["index_exists"] is True


async def _exercise_full_sync_rebuild(tmp_path, monkeypatch):
    index_path = tmp_path / "vector_index.faiss"
    monkeypatch.setattr(vector_store, "FAISS_INDEX_PATH", str(index_path))
    monkeypatch.setattr(vector_store.settings, "EMBEDDING_DIM", 4)

    embeddings = {
        "First text": np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        "Stale text": np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32),
        "Query": np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
    }

    store = VectorStoreService(FakeEmbeddingService(embeddings))
    await store.add_vector(1, "First text")
    await store.add_vector(99, "Stale text")

    paper_service = FakePaperService([FakePaper(1, "First text")])
    await asyncio.wait_for(store.full_sync(paper_service), timeout=1)

    return store.get_contents()


def test_vector_store_full_sync_rebuild_removes_stale_entries(tmp_path, monkeypatch):
    contents = asyncio.run(_exercise_full_sync_rebuild(tmp_path, monkeypatch))

    assert contents == [1]


async def _exercise_full_sync_adds_multiple_missing_entries(tmp_path, monkeypatch):
    index_path = tmp_path / "vector_index.faiss"
    monkeypatch.setattr(vector_store, "FAISS_INDEX_PATH", str(index_path))
    monkeypatch.setattr(vector_store.settings, "EMBEDDING_DIM", 4)

    embeddings = {
        "First text": np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        "Second text": np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32),
        "Third text": np.array([[0.0, 0.0, 1.0, 0.0]], dtype=np.float32),
    }

    store = VectorStoreService(FakeEmbeddingService(embeddings))
    await store.add_vector(1, "First text")

    paper_service = FakePaperService(
        [
            FakePaper(1, "First text"),
            FakePaper(2, "Second text"),
            FakePaper(3, "Third text"),
        ]
    )
    await asyncio.wait_for(store.full_sync(paper_service), timeout=1)

    return store.get_contents()


def test_vector_store_full_sync_adds_multiple_missing_entries(tmp_path, monkeypatch):
    contents = asyncio.run(
        _exercise_full_sync_adds_multiple_missing_entries(tmp_path, monkeypatch)
    )

    assert contents == [1, 2, 3]


def test_embedding_service_embed_batch_preserves_one_row_per_text():
    service = EmbeddingService()
    service._model = FakeModel()

    vectors = service.embed_batch(["First text", "Second text"])

    assert vectors.shape == (2, 4)
    np.testing.assert_allclose(
        vectors,
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )
