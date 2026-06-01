import logging
import os
from pathlib import Path

from huggingface_hub import snapshot_download

from ..core.config import settings

logger = logging.getLogger(__name__)

OFFLINE_ENV_VARS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
FALSEY_ENV_VALUES = {"", "0", "false", "no", "off"}


def _env_flag_enabled(name: str) -> bool:
    value = os.getenv(name)
    return value is not None and value.strip().lower() not in FALSEY_ENV_VALUES


def _enabled_offline_flags() -> list[str]:
    return [name for name in OFFLINE_ENV_VARS if _env_flag_enabled(name)]


def _pooling_config_path(model_dir: Path) -> Path:
    path = model_dir / settings.EMBEDDING_POOLING_CONFIG_PATH
    if path.suffix:
        return path
    return path / "config.json"


def _required_model_paths(model_dir: Path) -> list[Path]:
    return [
        model_dir / settings.EMBEDDING_ONNX_FILE,
        model_dir / "config.json",
        _pooling_config_path(model_dir),
    ]


def ensure_model():
    model_dir = Path(settings.EMBEDDING_MODEL_PATH)
    missing_paths = [
        path for path in _required_model_paths(model_dir) if not path.exists()
    ]

    if not missing_paths:
        logger.info("Embedding model already exists at %s", model_dir)
        return

    offline_flags = _enabled_offline_flags()
    if offline_flags:
        missing = ", ".join(str(path) for path in missing_paths)
        raise RuntimeError(
            "Embedding model files are missing, but Hugging Face downloads are "
            f"disabled by {', '.join(offline_flags)}. Remove those Railway "
            "environment variables or provide the model files at "
            f"{model_dir}. Missing: {missing}"
        )

    model_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading embedding model %s to %s", settings.REPO_ID, model_dir)

    snapshot_download(
        repo_id=settings.REPO_ID,
        local_dir=model_dir,
        local_dir_use_symlinks=False,
    )

    missing_paths = [
        path for path in _required_model_paths(model_dir) if not path.exists()
    ]
    if missing_paths:
        missing = ", ".join(str(path) for path in missing_paths)
        raise RuntimeError(
            "Embedding model download finished, but required files are still "
            f"missing: {missing}"
        )

    logger.info("Model downloaded successfully to %s", model_dir)
