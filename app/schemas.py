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
# Votes vs. comment stance schemas (descriptive only)
# ---------------------------------------------------------------------------

VotesStanceSource = Literal["gold", "live", "model"]


class VotesBlock(BaseModel):
    """Frozen-snapshot vote counts for a contribution."""
    votes_pro: int | None
    votes_contra: int | None
    # Fraction of thumbs-up among directional votes, in [0, 1], or None.
    thumbs_up_share: float | None
    snapshot_utc: str | None


class CommentsBlock(BaseModel):
    """Rolled-up comment stance counts for a contribution."""
    source: VotesStanceSource
    model: str | None
    n_comments: int
    favor: int
    against: int
    neutral: int
    mixed: int
    # favor / (favor + against) as a fraction in [0, 1] (mixed excluded), or None.
    comment_favor_share: float | None
    # Live verification coverage (None for gold/model). A live category reflects
    # only a comment's verified pairs, so these make the partial coverage explicit.
    live_pairs_verified: int | None = None
    live_pairs_total: int | None = None
    live_comments_fully_verified: int | None = None
    live_comments_partially_verified: int | None = None


class ContributionVotesStanceOut(BaseModel):
    """The votes-vs-stance panel for a single contribution."""
    contribution_id: int
    votes: VotesBlock
    comments: CommentsBlock
    snapshot_utc: str | None


class VotesStanceCounts(BaseModel):
    """Comment-category counts for one overview row."""
    favor: int
    against: int
    neutral: int
    mixed: int


class VotesStanceOverviewRow(BaseModel):
    """One descriptive per-contribution row for the scatter and table."""
    contribution_id: int
    thumbs_up_share: float | None
    comment_favor_share: float | None
    n_comments: int
    votes_pro: int | None
    votes_contra: int | None
    counts: VotesStanceCounts
    # Live verification coverage (None for gold/model).
    live_pairs_verified: int | None = None
    live_pairs_total: int | None = None
    live_comments_fully_verified: int | None = None
    live_comments_partially_verified: int | None = None


class SnapshotInfoOut(BaseModel):
    """Export-level metadata for the votes-vs-stance view header."""
    snapshot_utc: str | None
    n_contributions: int
    available_models: list[str]
    default_model: str | None
    default_source: VotesStanceSource
    evaluators: list[str]
    note: str


# ---------------------------------------------------------------------------
# Votes-vs-stance overall summary schemas (gold only)
# ---------------------------------------------------------------------------


class OverallCI90(BaseModel):
    """The 90% interval of the pre-registered point difference."""
    low: float
    high: float
    equivalent: bool


class OverallPrimary(BaseModel):
    """The pre-registered headline figures, copied verbatim from the export."""
    favor: int
    against: int
    neutral: int
    mixed: int
    favor_share: float
    vote_pro: int
    vote_contra: int
    vote_share: float
    difference_pp: float
    ci90: OverallCI90


class NoClearStance(BaseModel):
    """Comments with no clear stance: neutral plus mixed."""
    total: int
    neutral: int
    mixed: int


class OverallSummaryOut(BaseModel):
    """The pre-registered thumbs-vs-comment-stance headline block."""
    pre_registered: bool
    primary: OverallPrimary
    n_comments: int
    no_clear_stance: NoClearStance
    n_contributions_with_votes: int
    interpretation_rules: str


class SharedCI90(BaseModel):
    """A 90% range for one exploratory shared-contributions figure."""
    low: float
    high: float


class SharedBlock(BaseModel):
    """One shared-contributions gap (pooled or unweighted) with its range."""
    difference_pp: float
    ci90: SharedCI90


class SharedContributionsOut(BaseModel):
    """The exploratory, not-pre-registered shared-contributions block."""
    pre_registered: bool
    note: str
    thumbs_side_contributions: int
    comment_side_contributions: int
    both_sides_contributions: int
    pooled: SharedBlock
    unweighted: SharedBlock


class VotesStanceSummaryOut(BaseModel):
    """The overall votes-vs-stance summary: pre-registered plus exploratory."""
    snapshot_utc: str | None
    overall: OverallSummaryOut | None
    shared: SharedContributionsOut | None


# ---------------------------------------------------------------------------
# Per-key-message agreement schemas (gold only)
# ---------------------------------------------------------------------------


class KeyMessageComment(BaseModel):
    """One citizen comment behind a key-message stance count."""
    comment_text: str


class KeyMessageStanceComments(BaseModel):
    """The comments behind each stance count for one key message."""
    favor: list[KeyMessageComment]
    against: list[KeyMessageComment]
    neutral: list[KeyMessageComment]


class ContributionKeyMessageOut(BaseModel):
    """Gold agreement/disagreement for one key message of a contribution."""
    key_message: str
    favor: int
    against: int
    neutral: int
    n_with_stance: int
    too_few: bool
    comments: KeyMessageStanceComments


class KeyMessageOverviewItem(BaseModel):
    """One ranked key message in the agreement/rejection overview."""
    contribution_id: int
    key_message: str
    favor: int
    against: int
    neutral: int
    n_with_stance: int
    too_few: bool


class KeyMessageOverviewOut(BaseModel):
    """The most agreed-with and most rejected gold key messages."""
    most_agreed: list[KeyMessageOverviewItem]
    most_rejected: list[KeyMessageOverviewItem]


# ---------------------------------------------------------------------------
# Reset schema
# ---------------------------------------------------------------------------

class ResetEvaluationsOut(BaseModel):
    deleted_key_message_evaluations: int
    deleted_stance_evaluations: int
    deleted_added_key_messages: int
