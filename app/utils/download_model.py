import logging
import os

import requests

from ..core.config import settings

logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(settings.EMBEDDING_MODEL_PATH, settings.EMBEDDING_ONNX_FILE)
HF_ONNX_URL = (
    "https://huggingface.co/sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2/resolve/main/onnx/model.onnx"
)


def model_exists() -> bool:
    return os.path.exists(MODEL_PATH)


def ensure_model() -> None:
    if model_exists():
        logger.info("Embedding model already exists at %s", MODEL_PATH)
        return

    logger.info(
        "Downloading sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 ONNX model"
    )

    os.makedirs(settings.EMBEDDING_MODEL_PATH, exist_ok=True)

    with requests.get(HF_ONNX_URL, stream=True, timeout=(10, 120)) as response:
        response.raise_for_status()
        with open(MODEL_PATH, "wb") as model_file:
            for chunk in response.iter_content(chunk_size=8192):
                model_file.write(chunk)

    logger.info("Model downloaded successfully to %s", MODEL_PATH)
