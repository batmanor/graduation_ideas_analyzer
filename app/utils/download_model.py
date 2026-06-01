import logging
from pathlib import Path

from huggingface_hub import snapshot_download

from ..core.config import settings

logger = logging.getLogger(__name__)

MODEL_DIR = Path(settings.EMBEDDING_MODEL_PATH)

MODEL_PATH = MODEL_DIR / settings.EMBEDDING_ONNX_FILE

MODEL_CONF = MODEL_DIR / "config.json"

HF_INT8_URL = (
    "https://huggingface.co/Mo-alhariri/"
    "paraphrase-multilingual-minilm-l12-v2-int8/"
    "resolve/main/model.int8.onnx"
)


def ensure_model():
    if MODEL_PATH.exists() and MODEL_CONF.exists():
        logger.info(
            f"----------------------------------\n{'modelexists'.center(len('----------------------------------'), ' ')}\n----------------------------------"
        )
        return

    logger.info("Downloading INT8 embedding model")

    snapshot_download(
        repo_id="Mo-alhariri/paraphrase-multilingual-minilm-l12-v2-int8",
        local_dir=MODEL_DIR,
        local_dir_use_symlinks=False,
    )

    logger.info("Model downloaded successfully to %s", MODEL_PATH)
