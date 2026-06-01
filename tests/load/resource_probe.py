from __future__ import annotations

import argparse
import asyncio
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import settings  # noqa: E402
from app.core.metrics import get_rss_mb, metrics  # noqa: E402
from app.services import vector_store  # noqa: E402
from app.services.embedding_service import EmbeddingService  # noqa: E402
from app.services.vector_store import VectorStoreService  # noqa: E402
from app.utils import download_model  # noqa: E402


PROBE_TEXT = (
    "A multilingual retrieval system for validating research ideas against "
    "existing papers."
)


def report(step: str, max_rss_mb: float) -> None:
    rss_mb = get_rss_mb()
    rss_display = "unknown" if rss_mb is None else f"{rss_mb:.3f} MB"
    print(f"{step:<36} rss={rss_display}")

    if rss_mb is not None and rss_mb > max_rss_mb:
        raise RuntimeError(
            f"RSS exceeded limit after {step}: {rss_mb:.3f} MB > {max_rss_mb:.3f} MB"
        )


async def run_probe(max_rss_mb: float) -> None:
    report("process_start", max_rss_mb)

    embedding_service = EmbeddingService()
    report("embedding_service_created", max_rss_mb)

    embedding_service.get_model()
    report("embedding_model_loaded", max_rss_mb)

    embedding_service.embed(PROBE_TEXT)
    report("first_embedding_completed", max_rss_mb)

    with tempfile.TemporaryDirectory() as tmp_dir:
        index_path = Path(tmp_dir) / "probe_vector_index.faiss"
        original_index_path = vector_store.FAISS_INDEX_PATH
        vector_store.FAISS_INDEX_PATH = str(index_path)
        try:
            store = VectorStoreService(embedding_service)
            report("faiss_store_loaded", max_rss_mb)

            await store.add_vector(1, PROBE_TEXT)
            report("faiss_first_add_completed", max_rss_mb)

            await store.search(PROBE_TEXT, top_k=1)
            report("faiss_first_search_completed", max_rss_mb)

            await store.persist()
            report("faiss_persist_completed", max_rss_mb)
        finally:
            vector_store.FAISS_INDEX_PATH = original_index_path

    print("\nTiming snapshot:")
    for name, timing in metrics.snapshot()["timings"].items():
        print(f"{name:<32} {timing}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure embedding model and FAISS RSS against a memory budget."
    )
    parser.add_argument(
        "--max-rss-mb",
        type=float,
        default=500.0,
        help="Fail if process RSS exceeds this value.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Override EMBEDDING_MODEL_PATH for this probe process.",
    )
    parser.add_argument(
        "--onnx-file",
        default=None,
        help="Override EMBEDDING_ONNX_FILE for this probe process.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.model_path:
        settings.EMBEDDING_MODEL_PATH = Path(args.model_path)
    if args.onnx_file:
        settings.EMBEDDING_ONNX_FILE = args.onnx_file
    download_model.MODEL_PATH = (
        Path(settings.EMBEDDING_MODEL_PATH) / settings.EMBEDDING_ONNX_FILE
    )

    print(f"model_path={settings.EMBEDDING_MODEL_PATH}")
    print(f"onnx_file={settings.EMBEDDING_ONNX_FILE}")
    print(f"memory_limit={args.max_rss_mb:.3f} MB\n")

    try:
        asyncio.run(run_probe(args.max_rss_mb))
    except Exception as exc:
        print(f"\nFAILED: {exc}", file=sys.stderr)
        return 1

    print("\nPASSED: RSS stayed within limit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
