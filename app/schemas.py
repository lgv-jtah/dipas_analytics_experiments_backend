"""Pydantic schemas for request/response validation."""

from datetime import datetime
from typing import Literal

KeyMessageType = Literal["Zustand", "Wunsch", "Problem", "Qualität"]

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Contribution / key-message read schemas
# ---------------------------------------------------------------------------

class ContributionOut(BaseModel):
    contribution_id: int
    contribution_content: str


class KeyMessageOut(BaseModel):
    contribution_id: int
    contribution_content: str
    key_message: str
    key_message_type: str
    related_sentences: list[str]
    key_message_explanation: str


class StanceOut(BaseModel):
    contribution_id: int
    key_message: str
    comment_text: str
    stance: str
    explanation: str


# ---------------------------------------------------------------------------
# Evaluation write schemas
# ---------------------------------------------------------------------------

Verdict = Literal["correct", "incorrect"]


class KeyMessageEvaluationIn(BaseModel):
    contribution_id: int
    key_message: str
    verdict: Verdict
    comment: str | None = Field(default=None, description="Optional free-text note")
    evaluator: str | None = Field(default=None, description="Evaluator identifier")
    suggested_key_message: str | None = Field(default=None, description="Evaluator's own proposed key message")
    suggested_sentence: str | None = Field(default=None, description="Sentence from the contribution the suggested key message relates to")


class StanceEvaluationIn(BaseModel):
    contribution_id: int
    key_message: str
    comment_text: str
    verdict: Verdict
    comment: str | None = Field(default=None, description="Optional free-text note")
    evaluator: str | None = Field(default=None, description="Evaluator identifier")


# ---------------------------------------------------------------------------
# Evaluation read schemas
# ---------------------------------------------------------------------------

class KeyMessageEvaluationOut(BaseModel):
    id: int
    contribution_id: int
    key_message: str
    verdict: Verdict
    comment: str | None
    evaluator: str | None
    suggested_key_message: str | None
    suggested_sentence: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class StanceEvaluationOut(BaseModel):
    id: int
    contribution_id: int
    key_message: str
    comment_text: str
    verdict: Verdict
    comment: str | None
    evaluator: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Added key message schemas
# ---------------------------------------------------------------------------

class AddedKeyMessageIn(BaseModel):
    contribution_id: int
    key_message: str
    key_message_type: KeyMessageType
    key_message_sentence: str | None = Field(default=None)
    evaluator: str | None = Field(default=None)


class AddedKeyMessageOut(BaseModel):
    id: int
    contribution_id: int
    key_message: str
    key_message_type: str
    key_message_sentence: str | None
    evaluator: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Stats schema
# ---------------------------------------------------------------------------

class EvaluationStats(BaseModel):
    total_key_messages: int
    evaluated_key_messages: int
    correct_key_messages: int
    incorrect_key_messages: int
    total_stances: int
    evaluated_stances: int
    correct_stances: int
    incorrect_stances: int


class TesterEvaluationStats(EvaluationStats):
    tester: str
