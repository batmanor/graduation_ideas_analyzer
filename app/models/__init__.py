from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


from .paper import Paper as Paper  # noqa: E402
