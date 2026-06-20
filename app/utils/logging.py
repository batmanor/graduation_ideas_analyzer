import logging

from ..core.config import settings


def configure_logging() -> None:
    logging.basicConfig(
        level=settings.LOG_LEVEL.upper(),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(logger_name).propagate = False

    if not settings.SQLALCHEMY_ECHO:
        logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
