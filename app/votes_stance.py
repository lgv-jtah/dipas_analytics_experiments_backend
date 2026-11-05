"""Loads the frozen votes-vs-stance export and derives the descriptive panels.

The export is produced offline by the analysis repo
(``votes_stance_export.py``) and copied into ``data_exports/votes_stance/``.
It holds three artifacts plus a completion marker:

* ``votes_stance_comments.parquet`` - one row per gold comment, with the
  rolled-up ``gold_category`` and one ``category__<model>`` column per
  evaluated model. The comment (identified by ``comment_id``, a hash of
  contribution id and comment text) is the roll-up unit because different
  models extract different key messages, so only the comment set joins across
  sources.
* ``votes_snapshot.parquet`` - a frozen vote snapshot (no live API calls).
* ``votes_stance_export.summary.json`` - provenance, available models and the
  snapshot timestamp.
* ``artifact_hashes.json`` - the completion marker; every artifact is verified
  against it on load.

Everything here is descriptive only: the per-contribution favor shares are NOT
part of the pre-registered vote-ratio equivalence test.
"""

import hashlib
import json
import os
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import pandas as pd

from app import data

# --------------------------------------------------------------------------
# Locations and constants
# --------------------------------------------------------------------------

# The export directory; overridable via env so tests point at a temp copy.
VOTE_STANCE_EXPORT_DIR = Path(
    os.environ.get("VOTE_STANCE_EXPORT_DIR")
    or Path(__file__).parent.parent / "data_exports" / "votes_stance"
)

COMMENTS_FILE = "votes_stance_comments.parquet"
KEY_MESSAGES_FILE = "votes_stance_key_messages.parquet"
VOTES_FILE = "votes_snapshot.parquet"
SUMMARY_FILE = "votes_stance_export.summary.json"
HASHES_FILE = "artifact_hashes.json"

# A key message needs at least this many comments carrying a clear stance
# (in favour or against) before its agreement/rejection counts are worth
# showing; below it the counts are too thin to read anything into.
MIN_COMMENTS_WITH_STANCE = 3

# The per-model comment-category column prefix written by the exporter.
CATEGORY_PREFIX = "category__"
# The three canonical directional/neutral labels.
IN_FAVOR, AGAINST, NEUTRAL = "in_favor", "against", "neutral"
# The default model to show when source=model (the user's chosen default).
DEFAULT_MODEL = "gptoss-120b"
# The default data source everywhere.
DEFAULT_SOURCE = "gold"
ALLOWED_SOURCES = ("gold", "live", "model")

# Maps the raw (German/English) stance vocabulary used in the loaded parquet
# and in corrected_label to the canonical labels. Mirrors the exporter's rule.
STANCE_NORMALIZATION = {
    "in favor": IN_FAVOR,
    "ablehnung": AGAINST,
    "neutral": NEUTRAL,
}

# Verbatim caveat shown in the UI and carried from the export summary.
DESCRIPTIVE_ONLY_NOTE = (
    "These per-contribution figures are descriptive only and are not part of "
    "the pre-registered vote-ratio test."
)


# --------------------------------------------------------------------------
# Id and label helpers (mirror the analysis repo exactly)
# --------------------------------------------------------------------------


def _pair_id(contribution_id: object, key_message: str, comment_text: str) -> str:
    """Deterministic pair id, mirroring dataset_builder.compute_pair_id.

    The SHA-256 of ``"<contribution_id>|<key_message>|<comment_text>"`` cut to
    16 hex characters, over the *stringified* contribution id. Used to look up
    the loaded model's label for a reviewed pair.
    """
    key = f"{contribution_id}|{key_message}|{comment_text}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _comment_id(contribution_id: object, comment_text: str) -> str:
    """Deterministic comment id, mirroring reaction_split_review.comment_id_for.

    The SHA-256 of ``"<contribution_id>|<comment_text>"`` cut to 16 hex
    characters, over the *stringified* contribution id, so live verdicts roll
    up onto the same comments as the frozen export.
    """
    key = f"{contribution_id}|{comment_text}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def category_from_labels(labels: set[str]) -> str:
    """Collapse a comment's pair labels into one category.

    Mirrors ``vote_ratio_hypothesis._category_from_labels`` exactly:
    ``mixed`` when both directions are present, otherwise ``in_favor`` or
    ``against`` for a single direction, else ``neutral``.
    """
    has_favor, has_against = IN_FAVOR in labels, AGAINST in labels
    if has_favor and has_against:
        return "mixed"
    if has_favor:
        return IN_FAVOR
    if has_against:
        return AGAINST
    return NEUTRAL


def rollup(labels_by_comment: pd.DataFrame) -> dict[str, str]:
    """Roll per-pair labels up to one category per comment.

    Args:
        labels_by_comment: Rows with ``comment_id`` and a non-null ``label``
            (one of the canonical labels), one row per labelled pair.

    Returns:
        Maps each comment_id to its category (in_favor/against/mixed/neutral).
    """
    categories: dict[str, str] = {}
    for comment_id, group in labels_by_comment.groupby("comment_id", sort=False):
        categories[comment_id] = category_from_labels(set(group["label"]))
    return categories


def favor_counts(categories) -> dict:
    """Count comment categories and the favor share (mixed excluded).

    Args:
        categories: Iterable of category strings.

    Returns:
        A dict with integer ``favor``/``against``/``neutral``/``mixed`` counts,
        ``n_comments`` (the total), and ``favor_share`` - favor / (favor +
        against) as a fraction in [0, 1], or None when that denominator is 0.
        Mixed comments are excluded from the favor share exactly as the
        pre-registered rule does.
    """
    cats = list(categories)
    favor = cats.count(IN_FAVOR)
    against = cats.count(AGAINST)
    neutral = cats.count(NEUTRAL)
    mixed = cats.count("mixed")
    denominator = favor + against
    return {
        "favor": favor,
        "against": against,
        "neutral": neutral,
        "mixed": mixed,
        "n_comments": favor + against + neutral + mixed,
        "favor_share": None if denominator == 0 else favor / denominator,
    }


# --------------------------------------------------------------------------
# Export loading (cached, hash-verified)
# --------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file's bytes, read in 64 KiB chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _load_export() -> dict:
    """Load and hash-verify the export directory once.

    Returns:
        A dict with ``comments`` (DataFrame, contribution_id as str), ``votes``
        (DataFrame, contribution_id as str), ``key_messages`` (DataFrame with
        contribution_id as str, or ``None`` for an older export that predates
        the key-message artifact), ``summary`` (dict) and ``models`` (sorted
        list of model names present in the comments table).

    Raises:
        FileNotFoundError: If the export directory or a required file is
            missing.
        ValueError: If the hashes file does not record a completed export.
        RuntimeError: If any artifact's hash differs from the recorded one.
    """
    directory = VOTE_STANCE_EXPORT_DIR
    hashes_path = directory / HASHES_FILE
    if not hashes_path.exists():
        raise FileNotFoundError(
            f"Votes/stance export is incomplete or missing: {hashes_path}"
        )
    recorded = json.loads(hashes_path.read_text(encoding="utf-8"))
    if not recorded.get("complete"):
        raise ValueError(f"{hashes_path} does not record a completed export")
    # Verify every recorded artifact before trusting any of it.
    for name, expected in recorded.get("files", {}).items():
        actual = _sha256_file(directory / name)
        if actual != expected:
            raise RuntimeError(
                f"{directory / name} changed since it was written "
                f"(sha256 {actual}, recorded {expected})"
            )

    comments = pd.read_parquet(directory / COMMENTS_FILE)
    comments["contribution_id"] = comments["contribution_id"].astype(str)
    votes = pd.read_parquet(directory / VOTES_FILE)
    votes["contribution_id"] = votes["contribution_id"].astype(str)
    summary = json.loads((directory / SUMMARY_FILE).read_text(encoding="utf-8"))

    # The per-key-message artifact is optional: an export produced before this
    # feature existed simply does not have it, and the view degrades to empty
    # rather than failing to load the whole export.
    key_messages_path = directory / KEY_MESSAGES_FILE
    if key_messages_path.exists():
        key_messages = pd.read_parquet(key_messages_path)
        key_messages["contribution_id"] = key_messages["contribution_id"].astype(str)
    else:
        key_messages = None

    models = sorted(
        col[len(CATEGORY_PREFIX):]
        for col in comments.columns
        if col.startswith(CATEGORY_PREFIX)
    )
    return {
        "comments": comments,
        "votes": votes,
        "key_messages": key_messages,
        "summary": summary,
        "models": models,
    }


def available_models() -> list[str]:
    """Return the model names offered by the export, sorted."""
    return list(_load_export()["models"])


def snapshot_info() -> dict:
    """Return the export-level metadata for the UI header.

    Returns:
        A dict with ``snapshot_utc``, ``n_contributions``, ``available_models``,
        ``default_model`` and ``note``. The caller adds the evaluator list from
        the database.
    """
    export = _load_export()
    summary = export["summary"]
    models = export["models"]
    # The default model is the user's choice when present, else the first one.
    default_model = DEFAULT_MODEL if DEFAULT_MODEL in models else (models[0] if models else None)
    return {
        "snapshot_utc": summary.get("snapshot_utc"),
        "n_contributions": int(export["comments"]["contribution_id"].nunique()),
        "available_models": models,
        "default_model": default_model,
        "note": summary.get("descriptive_only_note", DESCRIPTIVE_ONLY_NOTE),
    }


# --------------------------------------------------------------------------
# Live labels (verdicts against the loaded model)
# --------------------------------------------------------------------------


def _loaded_model_index() -> tuple[dict[str, str], dict[str, set[str]]]:
    """Index the currently loaded parquet for live resolution and coverage.

    This is the model the annotators actually reviewed.

    Returns:
        A tuple ``(labels, pairs_by_comment)``. ``labels`` maps each pair_id to
        the loaded model's normalised stance (a "correct" verdict resolves to
        this). ``pairs_by_comment`` maps each comment_id to the set of the
        loaded model's pair_ids for that comment - the denominator for how many
        of a comment's pairs a human could verify.
    """
    df = data.get_dataframe()
    labels: dict[str, str] = {}
    pairs_by_comment: dict[str, set[str]] = defaultdict(set)
    for cid, km, comment, stance in zip(
        df["contribution_id"], df["key_message"], df["comment_text"], df["stance"]
    ):
        pair_id = _pair_id(cid, km, comment)
        pairs_by_comment[_comment_id(cid, comment)].add(pair_id)
        normalized = STANCE_NORMALIZATION.get(stance)
        if normalized is not None:
            labels[pair_id] = normalized
    return labels, pairs_by_comment


def _resolved_pair_labels(
    db, evaluator: str | None, loaded_labels: dict[str, str]
) -> dict[str, tuple[str, str]]:
    """Resolve stored verdicts to live labels, keyed by pair.

    Resolution rule (the user's decision):

    * ``correct`` -> the label the annotator reviewed, i.e. the loaded model's
      stance for that pair.
    * ``incorrect`` with a ``corrected_label`` -> the corrected label.
    * ``incorrect`` without a ``corrected_label`` -> dropped as unresolved.

    Only *verified* pairs are returned: a pair with no (resolvable) verdict is
    absent. A comment is therefore only ever partially covered by its verified
    pairs; see :func:`live_comment_resolution` for how a comment rolled up from
    a subset of its pairs is flagged as partially verified rather than
    presented as a full category.

    When no evaluator filter is given and several annotators evaluated the same
    pair, the most recently created verdict wins (rows are applied oldest
    first).

    Args:
        db: A SQLAlchemy session.
        evaluator: Restrict to one evaluator, or None for all.
        loaded_labels: pair_id -> loaded-model label (from
            :func:`_loaded_model_index`).

    Returns:
        Maps pair_id to ``(comment_id, label)`` for resolved pairs only.
    """
    # Imported here so the module loads without the DB models in pure-data use.
    from app.models import StanceEvaluation

    query = db.query(StanceEvaluation)
    if evaluator is not None:
        query = query.filter(StanceEvaluation.evaluator == evaluator)
    # Oldest first so a later annotator's verdict overwrites an earlier one.
    rows = query.order_by(StanceEvaluation.created_at.asc()).all()

    resolved: dict[str, tuple[str, str]] = {}
    for row in rows:
        pair_id = _pair_id(row.contribution_id, row.key_message, row.comment_text)
        comment_id = _comment_id(row.contribution_id, row.comment_text)
        if row.verdict == "correct":
            label = loaded_labels.get(pair_id)
            if label is None:
                # The reviewed pair is not in the loaded parquet; cannot resolve.
                continue
            resolved[pair_id] = (comment_id, label)
        elif row.verdict == "incorrect" and row.corrected_label:
            label = STANCE_NORMALIZATION.get(row.corrected_label)
            if label is not None:
                resolved[pair_id] = (comment_id, label)
        # incorrect without corrected_label -> dropped (unresolved).
    return resolved


def live_comment_resolution(db, evaluator: str | None = None,
                            universe: set[str] | None = None) -> dict:
    """Roll verified verdicts up per comment, tracking verification coverage.

    IMPORTANT: a live comment category is computed from only that comment's
    *verified* pairs. When a comment has more loaded-model pairs than have been
    verified, its category reflects the verified subset and may not represent
    the whole comment; such a comment is reported as partially verified so the
    partial coverage is never silent.

    Args:
        db: A SQLAlchemy session.
        evaluator: Restrict to one evaluator, or None for all.
        universe: If given, keep only these comment ids (the frozen export's
            comments), so live stays on the same comment set as gold/model, and
            coverage totals are computed over exactly this comment set.

    Returns:
        A dict with:
        * ``categories``: {comment_id: category} for comments with >=1 verified
          pair (possibly partially verified).
        * ``verified``: {comment_id: number of verified pairs}.
        * ``total``: {comment_id: number of loaded-model pairs} for every
          comment in ``universe`` (the verifiable denominator).
    """
    loaded_labels, pairs_by_comment = _loaded_model_index()
    resolved = _resolved_pair_labels(db, evaluator, loaded_labels)

    # Group the verified pair labels by comment.
    verified_labels: dict[str, dict[str, str]] = defaultdict(dict)
    for pair_id, (comment_id, label) in resolved.items():
        if universe is None or comment_id in universe:
            verified_labels[comment_id][pair_id] = label

    categories = {
        comment_id: category_from_labels(set(pair_map.values()))
        for comment_id, pair_map in verified_labels.items()
    }
    verified = {comment_id: len(pair_map) for comment_id, pair_map in verified_labels.items()}

    total: dict[str, int] = {}
    comment_ids = universe if universe is not None else set(pairs_by_comment)
    for comment_id in comment_ids:
        # A comment verified but absent from the loaded model still counts its
        # verified pairs, so it is never reported as "more verified than exist".
        total[comment_id] = max(
            len(pairs_by_comment.get(comment_id, ())), verified.get(comment_id, 0)
        )
    return {"categories": categories, "verified": verified, "total": total}


def live_comment_categories(db, evaluator: str | None = None,
                            universe: set[str] | None = None) -> dict[str, str]:
    """Return only the per-comment live categories (see live_comment_resolution).

    Each category reflects the comment's verified pairs only and may not
    represent the whole comment; callers that need coverage use
    :func:`live_comment_resolution` directly.
    """
    return live_comment_resolution(db, evaluator, universe)["categories"]


# --------------------------------------------------------------------------
# Per-pair label selection per source
# --------------------------------------------------------------------------


def _validate_source_model(source: str, model: str | None) -> None:
    """Allow-list the source and, for source=model, the model name.

    (Evaluator allow-listing is done in the route layer, where the database -
    the source of truth for valid evaluator names - is available.)

    Raises:
        ValueError: On an unknown source or (for source=model) unknown model.
    """
    if source not in ALLOWED_SOURCES:
        raise ValueError(f"Unknown source {source!r}; allowed: {ALLOWED_SOURCES}")
    if source == "model":
        models = available_models()
        if model not in models:
            raise ValueError(f"Unknown model {model!r}; allowed: {models}")


def _comment_categories(export: dict, source: str, model: str | None,
                        db, evaluator: str | None) -> tuple[pd.DataFrame, dict | None]:
    """Return per-comment categories for the source, plus live coverage.

    Comments with no category under the chosen source (a comment a model never
    labelled, or a comment with no verified live verdict) are dropped, so each
    source's favor share is computed only over the comments it actually labels.

    Returns:
        A tuple ``(categories_df, coverage)``. ``categories_df`` has
        ``contribution_id``/``comment_id``/``category`` for labelled comments.
        ``coverage`` is ``None`` for gold/model; for live it is the
        :func:`live_comment_resolution` result (``verified``/``total`` maps),
        used to report per-contribution verification coverage.
    """
    comments = export["comments"]
    out = comments[["contribution_id", "comment_id"]].copy()
    coverage: dict | None = None
    if source == "gold":
        out["category"] = comments["gold_category"].to_numpy()
    elif source == "model":
        out["category"] = comments[f"{CATEGORY_PREFIX}{model}"].to_numpy()
    else:  # live
        universe = set(comments["comment_id"])
        coverage = live_comment_resolution(db, evaluator, universe=universe)
        out["category"] = out["comment_id"].map(coverage["categories"])
    return out[out["category"].notna()], coverage


def _live_coverage(comment_ids, coverage: dict | None) -> dict:
    """Aggregate live verification coverage over a set of gold comments.

    Args:
        comment_ids: The gold comment ids of one contribution (or all).
        coverage: The :func:`live_comment_resolution` result, or None for a
            non-live source.

    Returns:
        For live: ``live_pairs_verified``, ``live_pairs_total``,
        ``live_comments_fully_verified`` and ``live_comments_partially_verified``.
        For a non-live source: all four are None (the figures do not apply).
    """
    if coverage is None:
        return {
            "live_pairs_verified": None,
            "live_pairs_total": None,
            "live_comments_fully_verified": None,
            "live_comments_partially_verified": None,
        }
    verified_map, total_map = coverage["verified"], coverage["total"]
    pairs_verified = pairs_total = fully = partially = 0
    for comment_id in comment_ids:
        verified = verified_map.get(comment_id, 0)
        total = total_map.get(comment_id, 0)
        pairs_verified += verified
        pairs_total += total
        if total > 0 and verified >= total:
            fully += 1
        elif verified > 0:
            partially += 1
    return {
        "live_pairs_verified": pairs_verified,
        "live_pairs_total": pairs_total,
        "live_comments_fully_verified": fully,
        "live_comments_partially_verified": partially,
    }


# --------------------------------------------------------------------------
# Panels
# --------------------------------------------------------------------------


def _votes_for(votes: pd.DataFrame, contribution_id: str) -> dict:
    """Return the vote block for one contribution (empty when no snapshot row)."""
    row = votes[votes["contribution_id"] == contribution_id]
    if row.empty:
        return {"votes_pro": None, "votes_contra": None, "thumbs_up_share": None}
    pro = int(row["votingPro"].iloc[0])
    contra = int(row["votingContra"].iloc[0])
    total = pro + contra
    return {
        "votes_pro": pro,
        "votes_contra": contra,
        "thumbs_up_share": None if total == 0 else pro / total,
    }


def contribution_panel(contribution_id: int, source: str = DEFAULT_SOURCE,
                       model: str | None = None, db=None,
                       evaluator: str | None = None) -> dict:
    """Build the votes-vs-stance panel for one contribution.

    Args:
        contribution_id: The contribution to describe.
        source: ``gold`` | ``live`` | ``model``.
        model: The model name (required and validated when source=model).
        db: A SQLAlchemy session, required when source=live.
        evaluator: Restrict live data to one evaluator, or None for all.

    Returns:
        A dict with ``contribution_id``, a ``votes`` block, a ``comments``
        block (counts + favor share + source/model) and ``snapshot_utc``.

    Raises:
        ValueError: On an unknown source/model.
        KeyError: If the contribution is not in the export.
    """
    _validate_source_model(source, model)
    export = _load_export()
    cid = str(contribution_id)
    if cid not in set(export["comments"]["contribution_id"]):
        raise KeyError(contribution_id)

    categories, coverage = _comment_categories(export, source, model, db, evaluator)
    subset = categories[categories["contribution_id"] == cid]
    counts = favor_counts(subset["category"])
    # Coverage is over this contribution's gold comments (every comment, even
    # those with no verified verdict, so the denominator is honest).
    contribution_comments = export["comments"]
    contribution_comments = contribution_comments[
        contribution_comments["contribution_id"] == cid
    ]["comment_id"]
    live_coverage = _live_coverage(contribution_comments, coverage)

    votes_block = _votes_for(export["votes"], cid)
    return {
        "contribution_id": int(contribution_id),
        "votes": {**votes_block, "snapshot_utc": export["summary"].get("snapshot_utc")},
        "comments": {
            "source": source,
            "model": model if source == "model" else None,
            "n_comments": counts["n_comments"],
            "favor": counts["favor"],
            "against": counts["against"],
            "neutral": counts["neutral"],
            "mixed": counts["mixed"],
            "comment_favor_share": counts["favor_share"],
            **live_coverage,
        },
        "snapshot_utc": export["summary"].get("snapshot_utc"),
    }


def overview(source: str = DEFAULT_SOURCE, model: str | None = None, db=None,
             evaluator: str | None = None) -> list[dict]:
    """Build the per-contribution overview rows for the scatter and table.

    Args:
        source: ``gold`` | ``live`` | ``model``.
        model: The model name (required when source=model).
        db: A SQLAlchemy session, required when source=live.
        evaluator: Restrict live data to one evaluator, or None for all.

    Returns:
        One row per contribution with ``contribution_id``,
        ``thumbs_up_share``, ``comment_favor_share``, ``n_comments``,
        ``votes_pro``, ``votes_contra`` and ``counts``
        (favor/against/neutral/mixed). Rows are sorted by contribution id.
    """
    _validate_source_model(source, model)
    export = _load_export()
    comments = export["comments"]
    votes = export["votes"]
    categories, coverage = _comment_categories(export, source, model, db, evaluator)
    # Map each contribution to its labelled-comment categories (may be empty).
    by_contribution = {
        cid: group["category"]
        for cid, group in categories.groupby("contribution_id", sort=False)
    }
    # Each contribution's full gold comment set (the coverage denominator).
    comments_by_contribution = {
        cid: group["comment_id"]
        for cid, group in comments.groupby("contribution_id", sort=False)
    }

    rows = []
    # Iterate over every contribution in the universe, even those with no
    # labelled comments under this source (they still get a row, with zeros).
    for cid in sorted(set(comments["contribution_id"]), key=int):
        counts = favor_counts(by_contribution.get(cid, []))
        votes_block = _votes_for(votes, cid)
        live_coverage = _live_coverage(comments_by_contribution.get(cid, []), coverage)
        rows.append(
            {
                "contribution_id": int(cid),
                "thumbs_up_share": votes_block["thumbs_up_share"],
                "comment_favor_share": counts["favor_share"],
                "n_comments": counts["n_comments"],
                "votes_pro": votes_block["votes_pro"],
                "votes_contra": votes_block["votes_contra"],
                "counts": {
                    "favor": counts["favor"],
                    "against": counts["against"],
                    "neutral": counts["neutral"],
                    "mixed": counts["mixed"],
                },
                **live_coverage,
            }
        )
    return rows


# --------------------------------------------------------------------------
# Overall summary blocks (gold only; read straight from the export summary)
# --------------------------------------------------------------------------


def overall_summary_block() -> dict | None:
    """Return the PRE-REGISTERED overall headline block from the export summary.

    This is the frozen vote-ratio headline the exporter copied verbatim (favor
    vs. thumbs-up share, the point difference and its 90% interval). It is
    returned as-is and never recomputed here.

    Returns:
        The ``overall_summary`` block, or ``None`` for an older export that does
        not carry it.
    """
    return _load_export()["summary"].get("overall_summary")


def shared_contributions_block() -> dict | None:
    """Return the EXPLORATORY shared-contributions block from the export summary.

    This block restricts both sides to the same contributions and is explicitly
    not pre-registered (it carries its own ``pre_registered: false`` flag and a
    note). It is returned verbatim from the export.

    Returns:
        The ``shared_contributions`` block, or ``None`` for an older export that
        does not carry it.
    """
    return _load_export()["summary"].get("shared_contributions")


# --------------------------------------------------------------------------
# Per-key-message agreement / disagreement (gold labels only)
# --------------------------------------------------------------------------


def _comment_texts(rows, label: str) -> list[dict]:
    """Collect the comment texts of one label for a key-message group.

    Args:
        rows: The gold key-message rows of a single key message, in their
            stored (first-appearance) order.
        label: The canonical label to filter on (in_favor/against/neutral).

    Returns:
        A list of ``{"comment_text": ...}`` dicts for the matching rows, in
        order; these are the comments behind that label's count.
    """
    return [
        {"comment_text": row.comment_text}
        for row in rows.itertuples(index=False)
        if row.reference_label == label
    ]


def _key_message_counts(rows) -> dict:
    """Count a key message's gold stance labels across its comments.

    Args:
        rows: The gold key-message rows of a single key message.

    Returns:
        A dict with integer ``favor``/``against``/``neutral`` counts, the
        directional total ``n_with_stance`` (favor + against) and ``too_few``
        (True when fewer than :data:`MIN_COMMENTS_WITH_STANCE` comments carry a
        clear stance).
    """
    labels = list(rows["reference_label"])
    favor = labels.count(IN_FAVOR)
    against = labels.count(AGAINST)
    neutral = labels.count(NEUTRAL)
    n_with_stance = favor + against
    return {
        "favor": favor,
        "against": against,
        "neutral": neutral,
        "n_with_stance": n_with_stance,
        "too_few": n_with_stance < MIN_COMMENTS_WITH_STANCE,
    }


def contribution_key_messages(contribution_id: int) -> list[dict]:
    """Per-key-message gold agreement/disagreement for one contribution.

    Each gold key message of the contribution is returned in first-appearance
    order with its in-favour / against / neutral comment counts and the comment
    texts behind each count, so the UI can show what comments express agreement
    or disagreement with that key message. Hand (gold) labels only.

    Args:
        contribution_id: The contribution to describe.

    Returns:
        One dict per gold key message with ``key_message``, the ``favor`` /
        ``against`` / ``neutral`` counts, ``n_with_stance``, ``too_few`` and a
        ``comments`` block listing the comment texts per label. Returns an empty
        list when the key-message artifact is absent (an older export).

    Raises:
        KeyError: If the artifact is present but the contribution is unknown.
    """
    key_messages = _load_export()["key_messages"]
    # Older export without the artifact: nothing to show, but not an error.
    if key_messages is None:
        return []
    cid = str(contribution_id)
    contribution_rows = key_messages[key_messages["contribution_id"] == cid]
    # A contribution with no gold key-message rows is genuinely unknown here.
    if contribution_rows.empty:
        raise KeyError(contribution_id)

    results: list[dict] = []
    # groupby(sort=False) preserves the first-appearance order of the key
    # messages, which the exporter deliberately froze into the parquet.
    for key_message, rows in contribution_rows.groupby("key_message", sort=False):
        counts = _key_message_counts(rows)
        results.append(
            {
                "key_message": key_message,
                **counts,
                "comments": {
                    "favor": _comment_texts(rows, IN_FAVOR),
                    "against": _comment_texts(rows, AGAINST),
                    "neutral": _comment_texts(rows, NEUTRAL),
                },
            }
        )
    return results


def _overview_items() -> list[dict]:
    """Build one ranked item per gold (contribution, key message) pair.

    Each key message is counted on its own; key messages are never grouped
    across contributions. Only the counts the overview ranks on are produced.

    Returns:
        A list of ``{contribution_id, key_message, favor, against, neutral,
        n_with_stance, too_few}`` dicts, one per (contribution, key message).
    """
    key_messages = _load_export()["key_messages"]
    if key_messages is None:
        return []
    items: list[dict] = []
    # Group by the (contribution, key message) pair so each key message is its
    # own item; sort=False keeps a stable, first-appearance traversal order.
    for (cid, key_message), rows in key_messages.groupby(
        ["contribution_id", "key_message"], sort=False
    ):
        counts = _key_message_counts(rows)
        items.append(
            {
                "contribution_id": int(cid),
                "key_message": key_message,
                **counts,
            }
        )
    return items


def key_message_overview(limit: int = 20) -> dict:
    """The most agreed-with and most rejected gold key messages.

    Both lists exclude key messages with fewer than
    :data:`MIN_COMMENTS_WITH_STANCE` comments carrying a clear stance, because a
    key message backed by one or two comments is too thin to rank. The lists
    report counts only, never percentages, and never group key messages across
    contributions. Hand (gold) labels only.

    Args:
        limit: Maximum number of items per list.

    Returns:
        A dict with ``most_agreed`` (key messages with at least one in-favour
        comment, most in-favour comments first) and ``most_rejected`` (at least
        one against comment, most against comments first).
    """
    # Only key messages with enough stance-bearing comments can be ranked.
    eligible = [item for item in _overview_items() if not item["too_few"]]

    def _share(item: dict, numerator: str) -> float:
        # The directional share used only to break ties between equal counts;
        # n_with_stance is >= MIN_COMMENTS_WITH_STANCE here, so it is never zero.
        return item[numerator] / item["n_with_stance"]

    # Most agreed: at least one in-favour comment, ranked by in-favour count,
    # then by favour share, then stably by contribution id and key message so
    # the order is fully deterministic.
    most_agreed = sorted(
        (item for item in eligible if item["favor"] >= 1),
        key=lambda item: (
            -item["favor"],
            -_share(item, "favor"),
            item["contribution_id"],
            item["key_message"],
        ),
    )[:limit]

    # Most rejected: at least one against comment, ranked the mirror way.
    most_rejected = sorted(
        (item for item in eligible if item["against"] >= 1),
        key=lambda item: (
            -item["against"],
            -_share(item, "against"),
            item["contribution_id"],
            item["key_message"],
        ),
    )[:limit]

    return {"most_agreed": most_agreed, "most_rejected": most_rejected}
