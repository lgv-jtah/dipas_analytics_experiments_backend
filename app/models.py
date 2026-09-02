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

    # Relationship to assigned stances
    stances: Mapped[list["AddedStance"]] = relationship(
        "AddedStance",
        back_populates="added_key_message",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        UniqueConstraint(
            "contribution_id", "key_message", "evaluator", name="uq_added_km"
        ),
    )


class AddedStance(Base):
    """Stores evaluator-assigned stances for added key messages."""

    __tablename__ = "added_stances"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    added_key_message_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("added_key_messages.id", ondelete="CASCADE"),
        nullable=False, index=True
    )
    contribution_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    comment_text: Mapped[str] = mapped_column(Text, nullable=False)
    comment_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    stance_label: Mapped[str] = mapped_column(String(20), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    # Relationship back to parent key message
    added_key_message: Mapped["AddedKeyMessage"] = relationship(
        "AddedKeyMessage", back_populates="stances"
    )

    __table_args__ = (
        UniqueConstraint(
            "added_key_message_id", "comment_text", "comment_index",
            name="uq_added_stance"
        ),
    )


class StanceEvaluation(Base):
    """Stores whether a human evaluator judged a predicted stance as correct."""

    __tablename__ = "stance_evaluations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    contribution_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    key_message: Mapped[str] = mapped_column(Text, nullable=False)
    comment_text: Mapped[str] = mapped_column(Text, nullable=False)
    comment_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # 'correct' | 'incorrect'
    verdict: Mapped[str] = mapped_column(String(20), nullable=False)
    corrected_label: Mapped[str | None] = mapped_column(String(20), nullable=True)
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
            "comment_index",
            "evaluator",
            name="uq_stance_eval",
        ),
    )
