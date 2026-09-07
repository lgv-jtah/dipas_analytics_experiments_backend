"""Pydantic schemas for request/response validation."""

from datetime import datetime
from typing import Literal

KeyMessageType = Literal["Zustand", "Wunsch", "Problem", "Qualität"]
StanceLabel = Literal["in favor", "neutral", "ablehnung"]

from pydantic import BaseModel, Field, field_validator


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
    comment_index: int | None = Field(default=None, description="Position of comment within key message stances")
    verdict: Verdict
    corrected_label: StanceLabel | None = Field(default=None, description="Correct label when verdict is 'incorrect'")
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
    comment_index: int | None
    verdict: Verdict
    corrected_label: StanceLabel | None
    comment: str | None
    evaluator: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Added key message schemas
# ---------------------------------------------------------------------------

class AddedStanceIn(BaseModel):
    """Schema for stance within an added key message."""
    comment_text: str
    comment_index: int | None = None
    stance_label: StanceLabel


class AddedStanceOut(BaseModel):
    """Output schema for added stance."""
    id: int
    comment_text: str
    comment_index: int | None
    stance_label: str
    created_at: datetime

    model_config = {"from_attributes": True}


class AddedKeyMessageIn(BaseModel):
    contribution_id: int
    key_message: str
    key_message_type: KeyMessageType
    key_message_sentence: str | None = Field(default=None)
    evaluator: str | None = Field(default=None)
    stances: list[AddedStanceIn] = Field(
        default_factory=list,
        description="List of stances to assign to this key message (at least one required)"
    )

    @field_validator('stances')
    @classmethod
    def validate_stances(cls, v):
        """Require at least one stance."""
        if not v or len(v) == 0:
            raise ValueError("At least one stance must be provided")
        return v


class AddedKeyMessageOut(BaseModel):
    id: int
    contribution_id: int
    key_message: str
    key_message_type: str
    key_message_sentence: str | None
    evaluator: str | None
    stances: list[AddedStanceOut] = []
    created_at: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Combined evaluation export schema
# ---------------------------------------------------------------------------

class AllEvaluationsOut(BaseModel):
    key_message_evaluations: list[KeyMessageEvaluationOut]
    stance_evaluations: list[StanceEvaluationOut]
    added_key_messages: list[AddedKeyMessageOut]


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


class ContributionStatsOut(BaseModel):
    contribution_id: int
    key_messages_count: int
    stances_count: int
    evaluated_key_messages_count: int
    evaluated_stances_count: int
    correct_key_messages_count: int
    incorrect_key_messages_count: int
    correct_stances_count: int
    incorrect_stances_count: int
    added_key_messages_count: int
    added_key_messages_missing_stances_count: int


# ---------------------------------------------------------------------------
# Dataset info schema
# ---------------------------------------------------------------------------

class DatasetInfoOut(BaseModel):
    model_name: str
    dataset_filename: str


# ---------------------------------------------------------------------------
# Reset schema
# ---------------------------------------------------------------------------

class ResetEvaluationsOut(BaseModel):
    deleted_key_message_evaluations: int
    deleted_stance_evaluations: int
    deleted_added_key_messages: int
