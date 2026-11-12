"""Shared fixtures for the votes-vs-stance backend tests.

Everything here is synthetic and isolated:

* the export directory is a temp copy pointed at by monkeypatching
  ``votes_stance.VOTE_STANCE_EXPORT_DIR`` and clearing its cache;
* the database is a temp SQLite file with the real schema, injected through
  FastAPI's dependency override - the real ``evaluations.db`` is never touched;
* ``data.get_dataframe`` is replaced with a small in-memory frame so no real
  model parquet is loaded.
"""
import hashlib
import json

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import data, votes_stance
from app.database import Base, get_db
from app.models import StanceEvaluation  # noqa: F401 - register the table
from app.routes import router


# --------------------------------------------------------------------------
# Synthetic export
# --------------------------------------------------------------------------


def _comment_id(cid, comment):
    """Mirror votes_stance._comment_id so fixtures join to the module under test."""
    return hashlib.sha256(f"{cid}|{comment}".encode("utf-8")).hexdigest()[:16]


# Gold comment rows (comment-level categories): contribution 10
# (favor + against -> share 0.5) and contribution 20 (mixed comment C +
# neutral comment D -> share undefined). Each tuple is
# (contribution_id, comment_text, gold_category, gptoss_category, base_category).
# gptoss agrees with gold everywhere; base flips comment B to in_favor and
# rolls comment C to in_favor (it only labelled the in_favor pair).
_COMMENT_ROWS = [
    (10, "comment A", "in_favor", "in_favor", "in_favor"),
    (10, "comment B", "against", "against", "in_favor"),
    (20, "comment C", "mixed", "mixed", "in_favor"),
    (20, "comment D", "neutral", "neutral", "neutral"),
]

# Gold key-message rows (one row per gold pair = one comment under a key
# message). Each tuple is (contribution_id, key_message, comment_text,
# reference_label). Designed so:
# * km-10-a: 3 in favor + 1 against -> 4 stance comments, NOT too few;
# * km-10-b: 2 in favor + 1 against -> 3 stance comments, NOT too few (and a
#   lower against share than km-10-a, to exercise the overview tiebreak);
# * km-20-a: 1 against + 1 neutral -> 1 stance comment, too few.
_KEY_MESSAGE_ROWS = [
    (10, "km-10-a", "kmA comment 1", "in_favor"),
    (10, "km-10-a", "kmA comment 2", "in_favor"),
    (10, "km-10-a", "kmA comment 3", "in_favor"),
    (10, "km-10-a", "kmA comment 4", "against"),
    (10, "km-10-b", "kmB comment 1", "in_favor"),
    (10, "km-10-b", "kmB comment 2", "in_favor"),
    (10, "km-10-b", "kmB comment 3", "against"),
    (20, "km-20-a", "kmX comment 1", "against"),
    (20, "km-20-a", "kmX comment 2", "neutral"),
]

# Synthetic overall-summary blocks, deliberately NOT the real figures, so a
# route test proves they are surfaced verbatim from the export.
_OVERALL_SUMMARY = {
    "pre_registered": True,
    "primary": {
        "favor": 5,
        "against": 2,
        "neutral": 1,
        "mixed": 1,
        "favor_share": 71.4,
        "vote_pro": 40,
        "vote_contra": 10,
        "vote_share": 80.0,
        "difference_pp": -8.6,
        "ci90": {"low": -15.0, "high": -2.0, "equivalent": False},
    },
    "n_comments": 9,
    "no_clear_stance": {"total": 2, "neutral": 1, "mixed": 1},
    "n_contributions_with_votes": 4,
    "interpretation_rules": "SYNTHETIC RULES: not the real text.",
}
_SHARED_CONTRIBUTIONS = {
    "pre_registered": False,
    "note": "SYNTHETIC: this comparison was not pre-registered.",
    "thumbs_side_contributions": 4,
    "comment_side_contributions": 3,
    "both_sides_contributions": 2,
    "bootstrap": {"n": 10, "seed": 1},
    "pooled": {"difference_pp": -5.0, "ci90": {"low": -10.0, "high": 1.0}},
    "unweighted": {"difference_pp": -4.0, "ci90": {"low": -9.0, "high": 2.0}},
}


def _write_export(directory, snapshot_utc="20260102T030405060708Z",
                  include_key_messages=True, include_summary_blocks=True):
    """Write a complete, hash-marked synthetic export into ``directory``.

    Args:
        directory: Where to write the export.
        snapshot_utc: The snapshot timestamp to record.
        include_key_messages: Write the per-key-message artifact (set False to
            simulate an older export that predates it).
        include_summary_blocks: Embed the overall-summary and
            shared-contributions blocks (set False for an older export).
    """
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for cid, comment, gold, gptoss, base in _COMMENT_ROWS:
        rows.append(
            {
                "comment_id": _comment_id(cid, comment),
                "contribution_id": str(cid),
                "comment_text": comment,
                "gold_category": gold,
                "category__gptoss-120b": gptoss,
                "category__base": base,
            }
        )
    pairs = pd.DataFrame(rows)
    votes = pd.DataFrame(
        {
            "contribution_id": [10, 20],
            "project_id": ["p-x", "p-y"],
            "votingPro": [7, 2],
            "votingContra": [3, 8],
            "commentsNumber": [2, 2],
            "dateCreated": ["2022-01-01", "2022-02-02"],
        }
    )
    summary = {
        "schema_version": "votes_stance_export_v1",
        "snapshot_utc": snapshot_utc,
        "descriptive_only_note": votes_stance.DESCRIPTIVE_ONLY_NOTE,
        "available_models": ["base", "gptoss-120b"],
    }
    if include_summary_blocks:
        summary["overall_summary"] = _OVERALL_SUMMARY
        summary["shared_contributions"] = _SHARED_CONTRIBUTIONS

    pairs_path = directory / votes_stance.COMMENTS_FILE
    votes_path = directory / votes_stance.VOTES_FILE
    summary_path = directory / votes_stance.SUMMARY_FILE
    pairs.to_parquet(pairs_path, index=False)
    votes.to_parquet(votes_path, index=False)
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    artifacts = [pairs_path, votes_path, summary_path]

    if include_key_messages:
        key_message_rows = [
            {
                "contribution_id": str(cid),
                "key_message": key_message,
                "comment_id": _comment_id(cid, comment),
                "comment_text": comment,
                "reference_label": label,
            }
            for cid, key_message, comment, label in _KEY_MESSAGE_ROWS
        ]
        key_messages_path = directory / votes_stance.KEY_MESSAGES_FILE
        pd.DataFrame(key_message_rows).to_parquet(key_messages_path, index=False)
        artifacts.append(key_messages_path)

    # Completion marker written last, covering every artifact.
    files = {}
    for path in artifacts:
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (directory / votes_stance.HASHES_FILE).write_text(
        json.dumps({"complete": True, "files": files}), encoding="utf-8"
    )


@pytest.fixture
def export_dir(tmp_path, monkeypatch):
    """A temp export directory wired into the module under test."""
    directory = tmp_path / "votes_stance"
    _write_export(directory)
    monkeypatch.setattr(votes_stance, "VOTE_STANCE_EXPORT_DIR", directory)
    votes_stance._load_export.cache_clear()
    yield directory
    votes_stance._load_export.cache_clear()


@pytest.fixture
def legacy_export_dir(tmp_path, monkeypatch):
    """A temp export WITHOUT the key-message artifact or summary blocks.

    Simulates an export produced before this feature existed, so the
    backward-compatibility path (still loads, degrades to empty) can be tested.
    """
    directory = tmp_path / "votes_stance_legacy"
    _write_export(directory, include_key_messages=False, include_summary_blocks=False)
    monkeypatch.setattr(votes_stance, "VOTE_STANCE_EXPORT_DIR", directory)
    votes_stance._load_export.cache_clear()
    yield directory
    votes_stance._load_export.cache_clear()


@pytest.fixture
def loaded_model_df(monkeypatch):
    """Replace data.get_dataframe with a small in-memory frame.

    The loaded model mirrors gptoss here, so a "correct" verdict resolves to
    the gold label for these pairs.
    """
    df = pd.DataFrame(
        {
            "contribution_id": [10, 10, 20, 20, 20],
            "key_message": ["kmA", "kmB", "kmC1", "kmC2", "kmD"],
            "comment_text": [
                "comment A",
                "comment B",
                "comment C",
                "comment C",
                "comment D",
            ],
            "stance": ["in favor", "ablehnung", "in favor", "ablehnung", "neutral"],
        }
    )
    monkeypatch.setattr(data, "get_dataframe", lambda: df)
    return df


# --------------------------------------------------------------------------
# Temp database + API client
# --------------------------------------------------------------------------


@pytest.fixture
def db_session(tmp_path):
    """A SQLAlchemy session bound to a throwaway SQLite file."""
    engine = create_engine(
        f"sqlite:///{tmp_path / 'test_evaluations.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSession()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


@pytest.fixture
def client(db_session):
    """A TestClient whose get_db yields the throwaway session."""
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides[get_db] = lambda: db_session
    return TestClient(app)


def add_stance_evaluation(session, *, contribution_id, key_message, comment_text,
                          verdict, corrected_label=None, evaluator=None):
    """Insert one stance evaluation row and commit it."""
    row = StanceEvaluation(
        contribution_id=contribution_id,
        key_message=key_message,
        comment_text=comment_text,
        comment_index=0,
        verdict=verdict,
        corrected_label=corrected_label,
        evaluator=evaluator,
    )
    session.add(row)
    session.commit()
    return row
