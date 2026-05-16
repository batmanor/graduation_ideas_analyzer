from sqlalchemy import Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column
from app.models import Base


class Paper(Base):
    __tablename__ = "papers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    external_id: Mapped[int] = mapped_column(
        Integer, unique=True, nullable=False, index=True
    )

    title: Mapped[str] = mapped_column(String, nullable=False)
    abstract: Mapped[str] = mapped_column(String, nullable=False)
    keywords: Mapped[str] = mapped_column(String, nullable=True)

    created_at: Mapped[str] = mapped_column(String, nullable=False, default=func.now())
    last_updated: Mapped[str] = mapped_column(
        String, nullable=False, default=func.now()
    )
