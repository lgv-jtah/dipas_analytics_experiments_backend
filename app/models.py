"""SQLAlchemy ORM models for storing human evaluations."""

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class KeyMessageEvaluation(Base):
    """Stores whether a human evaluator judged a predicted key message as correct."""

    __tablename__ = "key_message_evaluations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    contribution_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    key_message: Mapped[str] = mapped_column(Text, nullable=False)
    # 'correct' | 'incorrect'
    verdict: Mapped[str] = mapped_column(String(20), nullable=False)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    evaluator: Mapped[str | None] = mapped_column(String(100), nullable=True)
    suggested_key_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    suggested_sentence: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    __table_args__ = (
        UniqueConstraint(
            "contribution_id", "key_message", "evaluator", name="uq_km_eval"
        ),
    )


class AddedKeyMessage(Base):
    """Stores evaluator-added key messages for a contribution."""

    __tablename__ = "added_key_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    contribution_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    key_message: Mapped[str] = mapped_column(Text, nullable=False)
    key_message_type: Mapped[str] = mapped_column(String(50), nullable=False)
    key_message_sentence: Mapped[str | None] = mapped_column(Text, nullable=True)
    evaluator: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    __table_args__ = (
        UniqueConstraint(
            "contribution_id", "key_message", "evaluator", name="uq_added_km"
        ),
    )


class StanceEvaluation(Base):
    """Stores whether a human evaluator judged a predicted stance as correct."""

    __tablename__ = "stance_evaluations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    contribution_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    key_message: Mapped[str] = mapped_column(Text, nullable=False)
    comment_text: Mapped[str] = mapped_column(Text, nullable=False)
    # 'correct' | 'incorrect'
    verdict: Mapped[str] = mapped_column(String(20), nullable=False)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    evaluator: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    __table_args__ = (
        UniqueConstraint(
            "contribution_id",
            "key_message",
            "comment_text",
            "evaluator",
            name="uq_stance_eval",
        ),
    )
