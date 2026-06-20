import logging
from pathlib import Path

from huggingface_hub import snapshot_download

from ..core.config import settings


logger = logging.getLogger(__name__)

def _required_model_paths(model_dir: Path) -> list[Path]:
    return [
        model_dir / settings.EMBEDDING_ONNX_FILE,
        model_dir / "config.json",
        model_dir / settings.EMBEDDING_POOLING_CONFIG_PATH / "config.json",
    ]

def model_exists(model_dir: Path) -> bool:
    return all(
        path.exists()
        for path in _required_model_paths(model_dir)
    )

def download_model(model_dir) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading embedding model %s to %s", settings.REPO_ID, model_dir)

    snapshot_download(
        repo_id=settings.REPO_ID,
        local_dir=model_dir,
        local_dir_use_symlinks=False,
    )

def get_model() -> None:
    model_dir = Path(settings.EMBEDDING_MODEL_PATH)

    if model_exists(model_dir):
        logger.info("Embedding model already exists at %s", model_dir)
        return

    download_model(model_dir)

    missing_paths = [path for path in _required_model_paths(model_dir) if not path.exists()]

    if missing_paths:
        raise RuntimeError(
            f"Embedding model download finished, but required files are still missing:{missing_paths}"
        )

    logger.info("Model downloaded successfully to %s", model_dir)
