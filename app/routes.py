"""FastAPI router for contributions and their predictions."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app import data
from app.database import get_db
from app.models import AddedKeyMessage, KeyMessageEvaluation, StanceEvaluation
from app.schemas import (
    AddedKeyMessageIn,
    AddedKeyMessageOut,
    ContributionOut,
    EvaluationStats,
    KeyMessageEvaluationIn,
    KeyMessageEvaluationOut,
    KeyMessageOut,
    StanceEvaluationIn,
    StanceEvaluationOut,
    StanceOut,
    TesterEvaluationStats,
)

router = APIRouter()


# ---------------------------------------------------------------------------
# Contributions
# ---------------------------------------------------------------------------


@router.get(
    "/contributions",
    response_model=list[ContributionOut],
    summary="List all contributions",
    tags=["contributions"],
)
def list_contributions():
    """Return a paginated list of all contributions in the dataset."""
    return data.get_contributions()


@router.get(
    "/contributions/{contribution_id}",
    response_model=ContributionOut,
    summary="Get a single contribution",
    tags=["contributions"],
)
def get_contribution(contribution_id: int):
    rows = [c for c in data.get_contributions() if c["contribution_id"] == contribution_id]
    if not rows:
        raise HTTPException(status_code=404, detail="Contribution not found")
    return rows[0]


# ---------------------------------------------------------------------------
# Key messages
# ---------------------------------------------------------------------------


@router.get(
    "/contributions/{contribution_id}/key-messages",
    response_model=list[KeyMessageOut],
    summary="Get predicted key messages for a contribution",
    tags=["key-messages"],
)
def list_key_messages(contribution_id: int):
    """
    Returns all key messages extracted from the given contribution.
    Each entry includes `key_message_type` and `key_message_explanation`.
    """
    rows = data.get_key_messages(contribution_id)
    if not rows:
        raise HTTPException(
            status_code=404,
            detail="Contribution not found or has no key messages",
        )
    return rows


# ---------------------------------------------------------------------------
# Stances
# ---------------------------------------------------------------------------


@router.get(
    "/contributions/{contribution_id}/key-messages/{key_message}/stances",
    response_model=list[StanceOut],
    summary="Get predicted stances for a key message",
    tags=["stances"],
)
def list_stances(contribution_id: int, key_message: str):
    """
    Returns all comments and their predicted stance
    (`in favor` | `neutral` | `ablehnung`) for a given key message.
    """
    rows = data.get_stances(contribution_id, key_message)
    if not rows:
        raise HTTPException(
            status_code=404,
            detail="No stances found for this contribution / key message combination",
        )
    return rows


# ---------------------------------------------------------------------------
# Evaluator-added key messages
# (registered before any parameterised key-message routes)
# ---------------------------------------------------------------------------


@router.post(
    "/evaluations/key-messages/added",
    response_model=AddedKeyMessageOut,
    status_code=201,
    summary="Save an evaluator-added key message",
    tags=["evaluations"],
)
def add_key_message(
    payload: AddedKeyMessageIn,
    db: Session = Depends(get_db),
):
    existing = (
        db.query(AddedKeyMessage)
        .filter(
            AddedKeyMessage.contribution_id == payload.contribution_id,
            AddedKeyMessage.key_message == payload.key_message,
            AddedKeyMessage.evaluator == payload.evaluator,
        )
        .first()
    )

    if existing:
        existing.key_message_type = payload.key_message_type
        existing.key_message_sentence = payload.key_message_sentence
        db.commit()
        db.refresh(existing)
        return existing

    record = AddedKeyMessage(
        contribution_id=payload.contribution_id,
        key_message=payload.key_message,
        key_message_type=payload.key_message_type,
        key_message_sentence=payload.key_message_sentence,
        evaluator=payload.evaluator,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


@router.get(
    "/evaluations/key-messages/added",
    response_model=list[AddedKeyMessageOut],
    summary="List evaluator-added key messages",
    tags=["evaluations"],
)
def list_added_key_messages(
    contribution_id: int = Query(..., description="Filter by contribution"),
    evaluator: str | None = Query(default=None, description="Filter by evaluator name"),
    db: Session = Depends(get_db),
):
    query = db.query(AddedKeyMessage).filter(
        AddedKeyMessage.contribution_id == contribution_id
    )
    if evaluator is not None:
        query = query.filter(AddedKeyMessage.evaluator == evaluator)
    return query.all()


# ---------------------------------------------------------------------------
# Key-message evaluations
# ---------------------------------------------------------------------------


@router.post(
    "/evaluations/key-messages",
    response_model=KeyMessageEvaluationOut,
    status_code=201,
    summary="Submit a key-message evaluation",
    tags=["evaluations"],
)
def submit_key_message_evaluation(
    payload: KeyMessageEvaluationIn,
    db: Session = Depends(get_db),
):
    """
    Record whether the predicted key message is **correct** or **incorrect**.

    If an evaluation from the same evaluator for the same
    (contribution_id, key_message) pair already exists it is **updated**.
    """
    existing = (
        db.query(KeyMessageEvaluation)
        .filter(
            KeyMessageEvaluation.contribution_id == payload.contribution_id,
            KeyMessageEvaluation.key_message == payload.key_message,
            KeyMessageEvaluation.evaluator == payload.evaluator,
        )
        .first()
    )

    if existing:
        existing.verdict = payload.verdict
        existing.comment = payload.comment
        existing.suggested_key_message = payload.suggested_key_message
        existing.suggested_sentence = payload.suggested_sentence
        db.commit()
        db.refresh(existing)
        return existing

    evaluation = KeyMessageEvaluation(
        contribution_id=payload.contribution_id,
        key_message=payload.key_message,
        verdict=payload.verdict,
        comment=payload.comment,
        evaluator=payload.evaluator,
        suggested_key_message=payload.suggested_key_message,
        suggested_sentence=payload.suggested_sentence,
    )
    db.add(evaluation)
    db.commit()
    db.refresh(evaluation)
    return evaluation


@router.get(
    "/evaluations/key-messages",
    response_model=list[KeyMessageEvaluationOut],
    summary="List key-message evaluations",
    tags=["evaluations"],
)
def list_key_message_evaluations(
    contribution_id: int | None = Query(default=None),
    evaluator: str | None = Query(default=None),
    db: Session = Depends(get_db),
):
    query = db.query(KeyMessageEvaluation)
    if contribution_id is not None:
        query = query.filter(KeyMessageEvaluation.contribution_id == contribution_id)
    if evaluator is not None:
        query = query.filter(KeyMessageEvaluation.evaluator == evaluator)
    return query.all()


# ---------------------------------------------------------------------------
# Stance evaluations
# ---------------------------------------------------------------------------


@router.post(
    "/evaluations/stances",
    response_model=StanceEvaluationOut,
    status_code=201,
    summary="Submit a stance evaluation",
    tags=["evaluations"],
)
def submit_stance_evaluation(
    payload: StanceEvaluationIn,
    db: Session = Depends(get_db),
):
    """
    Record whether the predicted stance for a comment is **correct** or **incorrect**.

    If an evaluation from the same evaluator for the same
    (contribution_id, key_message, comment_text) triple already exists it is **updated**.
    """
    existing = (
        db.query(StanceEvaluation)
        .filter(
            StanceEvaluation.contribution_id == payload.contribution_id,
            StanceEvaluation.key_message == payload.key_message,
            StanceEvaluation.comment_text == payload.comment_text,
            StanceEvaluation.evaluator == payload.evaluator,
        )
        .first()
    )

    if existing:
        existing.verdict = payload.verdict
        existing.comment = payload.comment
        db.commit()
        db.refresh(existing)
        return existing

    evaluation = StanceEvaluation(
        contribution_id=payload.contribution_id,
        key_message=payload.key_message,
        comment_text=payload.comment_text,
        verdict=payload.verdict,
        comment=payload.comment,
        evaluator=payload.evaluator,
    )
    db.add(evaluation)
    db.commit()
    db.refresh(evaluation)
    return evaluation


@router.get(
    "/evaluations/stances",
    response_model=list[StanceEvaluationOut],
    summary="List stance evaluations",
    tags=["evaluations"],
)
def list_stance_evaluations(
    contribution_id: int | None = Query(default=None),
    evaluator: str | None = Query(default=None),
    db: Session = Depends(get_db),
):
    query = db.query(StanceEvaluation)
    if contribution_id is not None:
        query = query.filter(StanceEvaluation.contribution_id == contribution_id)
    if evaluator is not None:
        query = query.filter(StanceEvaluation.evaluator == evaluator)
    return query.all()


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def _compute_stats(km_evals: list, stance_evals: list) -> dict:
    df = data.get_dataframe()
    return dict(
        total_key_messages=int(df.drop_duplicates(["contribution_id", "key_message"]).shape[0]),
        evaluated_key_messages=len(km_evals),
        correct_key_messages=sum(1 for e in km_evals if e.verdict == "correct"),
        incorrect_key_messages=sum(1 for e in km_evals if e.verdict == "incorrect"),
        total_stances=int(df.shape[0]),
        evaluated_stances=len(stance_evals),
        correct_stances=sum(1 for e in stance_evals if e.verdict == "correct"),
        incorrect_stances=sum(1 for e in stance_evals if e.verdict == "incorrect"),
    )


@router.get(
    "/evaluations/stats",
    response_model=EvaluationStats,
    summary="Overall evaluation progress",
    tags=["evaluations"],
)
def get_stats(db: Session = Depends(get_db)):
    """Summarises how many predictions have been evaluated so far across all testers."""
    km_evals = db.query(KeyMessageEvaluation).all()
    stance_evals = db.query(StanceEvaluation).all()
    return EvaluationStats(**_compute_stats(km_evals, stance_evals))


@router.get(
    "/evaluations/stats/{tester}",
    response_model=TesterEvaluationStats,
    summary="Evaluation progress for a single tester",
    tags=["evaluations"],
)
def get_stats_for_tester(tester: str, db: Session = Depends(get_db)):
    """Summarises how many predictions the given tester has evaluated so far."""
    km_evals = (
        db.query(KeyMessageEvaluation)
        .filter(KeyMessageEvaluation.evaluator == tester)
        .all()
    )
    stance_evals = (
        db.query(StanceEvaluation)
        .filter(StanceEvaluation.evaluator == tester)
        .all()
    )
    return TesterEvaluationStats(tester=tester, **_compute_stats(km_evals, stance_evals))
