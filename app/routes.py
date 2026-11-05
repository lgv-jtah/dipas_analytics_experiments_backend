"""FastAPI router for contributions and their predictions."""

from collections import defaultdict

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func
from sqlalchemy.orm import Session, joinedload

from app import data, votes_stance
from app.database import get_db
from app.models import AddedKeyMessage, AddedStance, KeyMessageEvaluation, StanceEvaluation
from app.schemas import (
    AddedKeyMessageIn,
    AddedKeyMessageOut,
    AllEvaluationsOut,
    ContributionKeyMessageOut,
    ContributionOut,
    ContributionStatsOut,
    ContributionVotesStanceOut,
    DatasetInfoOut,
    EvaluationStats,
    KeyMessageEvaluationIn,
    KeyMessageEvaluationOut,
    KeyMessageOut,
    KeyMessageOverviewOut,
    ResetEvaluationsOut,
    SnapshotInfoOut,
    StanceEvaluationIn,
    StanceEvaluationOut,
    StanceOut,
    TesterEvaluationStats,
    VotesStanceOverviewRow,
    VotesStanceSource,
    VotesStanceSummaryOut,
)

router = APIRouter()


# ---------------------------------------------------------------------------
# Dataset info
# ---------------------------------------------------------------------------


@router.get(
    "/dataset/model",
    response_model=DatasetInfoOut,
    summary="Which model's predictions are currently loaded",
    tags=["dataset"],
)
def get_dataset_info():
    return DatasetInfoOut(
        model_name=data.get_model_name(),
        dataset_filename=data.PARQUET_PATH.name,
    )


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
    "/contributions/{contribution_id}/stances",
    response_model=list[StanceOut],
    summary="Get predicted stances for a key message",
    tags=["stances"],
)
def list_stances(
    contribution_id: int,
    key_message: str = Query(..., description="Exact key message text"),
):
    """
    Returns all comments and their predicted stance
    (`in favor` | `neutral` | `ablehnung`) for a given key message.

    `key_message` is a query parameter (not a path segment) because key
    message text is free-form and may contain characters like `/` that
    can't be reliably round-tripped through a URL path segment.
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
    summary="Save an evaluator-added key message with stances",
    tags=["evaluations"],
)
def add_key_message(
    payload: AddedKeyMessageIn,
    db: Session = Depends(get_db),
):
    """
    Save a new key message identified by an evaluator, along with
    assigned stance-labeled comments.
    
    Requires at least one stance to be provided.
    """
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
        # Update existing key message
        existing.key_message_type = payload.key_message_type
        existing.key_message_sentence = payload.key_message_sentence
        
        # Delete old stances and replace with new ones
        db.query(AddedStance).filter(
            AddedStance.added_key_message_id == existing.id
        ).delete()
        
        # Add new stances
        for stance_data in payload.stances:
            stance = AddedStance(
                added_key_message_id=existing.id,
                contribution_id=payload.contribution_id,
                comment_text=stance_data.comment_text,
                comment_index=stance_data.comment_index,
                stance_label=stance_data.stance_label,
            )
            db.add(stance)
        
        db.commit()
        db.refresh(existing)
        return existing

    # Create new key message
    record = AddedKeyMessage(
        contribution_id=payload.contribution_id,
        key_message=payload.key_message,
        key_message_type=payload.key_message_type,
        key_message_sentence=payload.key_message_sentence,
        evaluator=payload.evaluator,
    )
    db.add(record)
    db.flush()  # Get the ID before creating stances

    # Create stances
    for stance_data in payload.stances:
        stance = AddedStance(
            added_key_message_id=record.id,
            contribution_id=payload.contribution_id,
            comment_text=stance_data.comment_text,
            comment_index=stance_data.comment_index,
            stance_label=stance_data.stance_label,
        )
        db.add(stance)

    db.commit()
    db.refresh(record)
    return record


@router.get(
    "/evaluations/key-messages/added",
    response_model=list[AddedKeyMessageOut],
    summary="List evaluator-added key messages with stances",
    tags=["evaluations"],
)
def list_added_key_messages(
    contribution_id: int = Query(..., description="Filter by contribution"),
    evaluator: str | None = Query(default=None, description="Filter by evaluator name"),
    db: Session = Depends(get_db),
):
    """Fetch added key messages with their assigned stances."""
    query = db.query(AddedKeyMessage).filter(
        AddedKeyMessage.contribution_id == contribution_id
    )
    if evaluator is not None:
        query = query.filter(AddedKeyMessage.evaluator == evaluator)
    
    # Eager load stances to avoid N+1 queries
    query = query.options(joinedload(AddedKeyMessage.stances))
    
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
    (contribution_id, key_message, comment_text, comment_index) triple already exists it is **updated**.

    Evaluations saved before comment_index tracking existed have
    comment_index = NULL. Since the frontend always sends a real index
    now, an exact match would never find those legacy rows - instead of
    silently creating a duplicate, we fall back to adopting the single
    unambiguous legacy row for this (contribution_id, key_message,
    comment_text, evaluator) combination, if one exists, and upgrade its
    comment_index to the incoming value.
    """
    # Validate: corrected_label is required when verdict is incorrect
    if payload.verdict == "incorrect" and not payload.corrected_label:
        raise HTTPException(
            status_code=400,
            detail="corrected_label is required when verdict is 'incorrect'"
        )
    
    existing = (
        db.query(StanceEvaluation)
        .filter(
            StanceEvaluation.contribution_id == payload.contribution_id,
            StanceEvaluation.key_message == payload.key_message,
            StanceEvaluation.comment_text == payload.comment_text,
            StanceEvaluation.comment_index == payload.comment_index,
            StanceEvaluation.evaluator == payload.evaluator,
        )
        .first()
    )

    if not existing and payload.comment_index is not None:
        legacy_candidates = (
            db.query(StanceEvaluation)
            .filter(
                StanceEvaluation.contribution_id == payload.contribution_id,
                StanceEvaluation.key_message == payload.key_message,
                StanceEvaluation.comment_text == payload.comment_text,
                StanceEvaluation.comment_index.is_(None),
                StanceEvaluation.evaluator == payload.evaluator,
            )
            .all()
        )
        if len(legacy_candidates) == 1:
            existing = legacy_candidates[0]
            existing.comment_index = payload.comment_index

    if existing:
        existing.verdict = payload.verdict
        existing.corrected_label = payload.corrected_label
        existing.comment = payload.comment
        db.commit()
        db.refresh(existing)
        return existing

    evaluation = StanceEvaluation(
        contribution_id=payload.contribution_id,
        key_message=payload.key_message,
        comment_text=payload.comment_text,
        comment_index=payload.comment_index,
        verdict=payload.verdict,
        corrected_label=payload.corrected_label,
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
# Combined evaluation export
# ---------------------------------------------------------------------------


@router.get(
    "/evaluations/all",
    response_model=AllEvaluationsOut,
    summary="Export all evaluation data",
    tags=["evaluations"],
)
def get_all_evaluations(
    evaluator: str | None = Query(
        default=None, description="Filter all results by evaluator name"
    ),
    db: Session = Depends(get_db),
):
    """Returns every key-message evaluation, stance evaluation, and
    evaluator-added key message, optionally filtered to a single evaluator."""
    km_query = db.query(KeyMessageEvaluation)
    stance_query = db.query(StanceEvaluation)
    added_query = db.query(AddedKeyMessage)

    if evaluator is not None:
        km_query = km_query.filter(KeyMessageEvaluation.evaluator == evaluator)
        stance_query = stance_query.filter(StanceEvaluation.evaluator == evaluator)
        added_query = added_query.filter(AddedKeyMessage.evaluator == evaluator)

    return AllEvaluationsOut(
        key_message_evaluations=km_query.all(),
        stance_evaluations=stance_query.all(),
        added_key_messages=added_query.all(),
    )


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


@router.get(
    "/evaluations/contributions-stats",
    response_model=list[ContributionStatsOut],
    summary="Per-contribution evaluation progress",
    tags=["evaluations"],
)
def get_contributions_stats(
    evaluator: str | None = Query(
        default=None, description="Restrict evaluation counts to a single evaluator"
    ),
    db: Session = Depends(get_db),
):
    """
    Breaks down dataset size and evaluation progress per contribution, so the
    frontend can show a per-contribution overview instead of only the
    dataset-wide totals from `/evaluations/stats`.

    `evaluated_key_messages_count` / `evaluated_stances_count` count distinct
    predictions that have at least one verdict (deduped across evaluators when
    no `evaluator` filter is given). `correct_*` / `incorrect_*` counts count
    evaluation records (matching `/evaluations/stats`'s semantics), so they
    can exceed the evaluated count if multiple evaluators disagreed on the
    same prediction.
    """
    dataset_counts = data.get_contribution_dataset_counts()

    km_query = db.query(KeyMessageEvaluation)
    stance_query = db.query(StanceEvaluation)
    added_query = db.query(AddedKeyMessage).options(joinedload(AddedKeyMessage.stances))
    if evaluator is not None:
        km_query = km_query.filter(KeyMessageEvaluation.evaluator == evaluator)
        stance_query = stance_query.filter(StanceEvaluation.evaluator == evaluator)
        added_query = added_query.filter(AddedKeyMessage.evaluator == evaluator)

    km_evaluated = defaultdict(set)
    km_correct = defaultdict(int)
    km_incorrect = defaultdict(int)
    for e in km_query.all():
        km_evaluated[e.contribution_id].add(e.key_message)
        if e.verdict == "correct":
            km_correct[e.contribution_id] += 1
        else:
            km_incorrect[e.contribution_id] += 1

    stance_evaluated = defaultdict(set)
    stance_correct = defaultdict(int)
    stance_incorrect = defaultdict(int)
    for e in stance_query.all():
        stance_evaluated[e.contribution_id].add((e.key_message, e.comment_text))
        if e.verdict == "correct":
            stance_correct[e.contribution_id] += 1
        else:
            stance_incorrect[e.contribution_id] += 1

    # Stance-labeling for an added key message is all-or-nothing (a single
    # save assigns a label to every comment at once), so "zero stances"
    # reliably means "not yet labeled" - no partial-completion tracking needed.
    added_km_count = defaultdict(int)
    added_km_missing_stances = defaultdict(int)
    for a in added_query.all():
        added_km_count[a.contribution_id] += 1
        if not a.stances:
            added_km_missing_stances[a.contribution_id] += 1

    return [
        ContributionStatsOut(
            contribution_id=cid,
            key_messages_count=counts["key_messages_count"],
            stances_count=counts["stances_count"],
            evaluated_key_messages_count=len(km_evaluated.get(cid, ())),
            evaluated_stances_count=len(stance_evaluated.get(cid, ())),
            correct_key_messages_count=km_correct.get(cid, 0),
            incorrect_key_messages_count=km_incorrect.get(cid, 0),
            correct_stances_count=stance_correct.get(cid, 0),
            incorrect_stances_count=stance_incorrect.get(cid, 0),
            added_key_messages_count=added_km_count.get(cid, 0),
            added_key_messages_missing_stances_count=added_km_missing_stances.get(cid, 0),
        )
        for cid, counts in sorted(dataset_counts.items())
    ]


# ---------------------------------------------------------------------------
# Votes vs. comment stance (descriptive only)
# ---------------------------------------------------------------------------


def _list_evaluators(db: Session) -> list[str]:
    """Return the distinct, non-empty evaluator names seen in stance evaluations."""
    rows = (
        db.query(func.distinct(StanceEvaluation.evaluator))
        .filter(StanceEvaluation.evaluator.isnot(None))
        .all()
    )
    return sorted(name for (name,) in rows if name)


def _check_evaluator(db: Session, evaluator: str | None) -> None:
    """Allow-list the evaluator query param against the known evaluators.

    Raises:
        HTTPException: 400 if a non-empty evaluator is not a known name.
    """
    if evaluator is not None and evaluator not in _list_evaluators(db):
        raise HTTPException(status_code=400, detail=f"Unknown evaluator {evaluator!r}")


@router.get(
    "/votes-stance/snapshot-info",
    response_model=SnapshotInfoOut,
    summary="Metadata for the votes-vs-stance view",
    tags=["votes-stance"],
)
def get_votes_stance_snapshot_info(db: Session = Depends(get_db)):
    """Return the frozen snapshot time, available models and evaluator list.

    The figures this view produces are descriptive only (see the returned
    ``note``) and are not part of the pre-registered vote-ratio test.
    """
    info = votes_stance.snapshot_info()
    return SnapshotInfoOut(
        **info,
        default_source=votes_stance.DEFAULT_SOURCE,
        evaluators=_list_evaluators(db),
    )


@router.get(
    "/contributions/{contribution_id}/votes-stance",
    response_model=ContributionVotesStanceOut,
    summary="Votes vs. comment stance for one contribution",
    tags=["votes-stance"],
)
def get_contribution_votes_stance(
    contribution_id: int,
    source: VotesStanceSource = Query(default=votes_stance.DEFAULT_SOURCE),
    model: str | None = Query(default=None),
    evaluator: str | None = Query(default=None),
    db: Session = Depends(get_db),
):
    """Return the vote counts and rolled-up comment stance for a contribution.

    `source` selects the comment labels: `gold` (frozen gold labels), `model`
    (a model's predictions; `model` query param required) or `live` (resolved
    from stored human verdicts against the loaded model). A `live` source with
    no data yields empty counts rather than an error.
    """
    _check_evaluator(db, evaluator)
    try:
        return votes_stance.contribution_panel(
            contribution_id, source=source, model=model, db=db, evaluator=evaluator
        )
    except ValueError as exc:
        # Unknown model (source is already constrained by the Literal type).
        raise HTTPException(status_code=400, detail=str(exc))
    except KeyError:
        raise HTTPException(status_code=404, detail="Contribution not found")


@router.get(
    "/votes-stance/overview",
    response_model=list[VotesStanceOverviewRow],
    summary="Per-contribution votes-vs-stance overview",
    tags=["votes-stance"],
)
def get_votes_stance_overview(
    source: VotesStanceSource = Query(default=votes_stance.DEFAULT_SOURCE),
    model: str | None = Query(default=None),
    evaluator: str | None = Query(default=None),
    db: Session = Depends(get_db),
):
    """Return one descriptive row per contribution for the scatter and table."""
    _check_evaluator(db, evaluator)
    try:
        return votes_stance.overview(
            source=source, model=model, db=db, evaluator=evaluator
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get(
    "/votes-stance/summary",
    response_model=VotesStanceSummaryOut,
    summary="Overall votes-vs-comment-stance summary (gold only)",
    tags=["votes-stance"],
)
def get_votes_stance_summary():
    """Return the overall summary: the pre-registered headline plus the
    exploratory shared-contributions block.

    This view is gold-only: it takes no source, model or evaluator parameters.
    The ``overall`` block is the pre-registered figure, copied verbatim from the
    frozen export; the ``shared`` block is explicitly not pre-registered (it
    carries its own flag and note). Either may be ``null`` for an older export
    that predates these blocks.
    """
    export = votes_stance._load_export()
    return VotesStanceSummaryOut(
        snapshot_utc=export["summary"].get("snapshot_utc"),
        overall=votes_stance.overall_summary_block(),
        shared=votes_stance.shared_contributions_block(),
    )


@router.get(
    "/contributions/{contribution_id}/votes-stance/key-messages",
    response_model=list[ContributionKeyMessageOut],
    summary="Per-key-message gold agreement/disagreement for a contribution",
    tags=["votes-stance"],
)
def get_contribution_key_messages(contribution_id: int):
    """Return the gold agreement/disagreement for each key message.

    This view is gold-only: it takes no source or model parameters. It lists one
    row per gold key message with in-favour / against / neutral comment counts
    and the comments behind them. An unknown contribution is a 404; when the
    key-message artifact is absent (an older export) the list is empty.
    """
    try:
        return votes_stance.contribution_key_messages(contribution_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Contribution not found")


@router.get(
    "/votes-stance/key-messages/overview",
    response_model=KeyMessageOverviewOut,
    summary="Most agreed-with and most rejected gold key messages",
    tags=["votes-stance"],
)
def get_key_message_overview(limit: int = Query(20, ge=1, le=100)):
    """Return the most agreed-with and most rejected gold key messages.

    This view is gold-only. Only key messages with at least three comments
    carrying a clear stance are listed; counts only, never percentages, and key
    messages are never grouped across contributions.
    """
    return votes_stance.key_message_overview(limit=limit)


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


@router.delete(
    "/evaluations/reset",
    response_model=ResetEvaluationsOut,
    summary="Delete all evaluation data",
    tags=["evaluations"],
)
def reset_evaluations(
    confirm: bool = Query(
        default=False,
        description="Must be set to true to actually perform the reset.",
    ),
    db: Session = Depends(get_db),
):
    """
    Empties the key-message evaluations, stance evaluations, and
    evaluator-added key messages tables. Use this after switching to a
    different model's parquet dataset so old verdicts don't get mixed up
    with evaluations of the new model's predictions.

    This does not touch the underlying parquet dataset.
    """
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Pass confirm=true to permanently delete all evaluation data.",
        )

    deleted_km = db.query(KeyMessageEvaluation).delete()
    deleted_stance = db.query(StanceEvaluation).delete()
    deleted_added = db.query(AddedKeyMessage).delete()
    db.commit()

    return ResetEvaluationsOut(
        deleted_key_message_evaluations=deleted_km,
        deleted_stance_evaluations=deleted_stance,
        deleted_added_key_messages=deleted_added,
    )
