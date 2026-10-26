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


def _write_export(directory, snapshot_utc="20260102T030405060708Z"):
    """Write a complete, hash-marked synthetic export into ``directory``."""
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

    pairs_path = directory / votes_stance.COMMENTS_FILE
    votes_path = directory / votes_stance.VOTES_FILE
    summary_path = directory / votes_stance.SUMMARY_FILE
    pairs.to_parquet(pairs_path, index=False)
    votes.to_parquet(votes_path, index=False)
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    # Completion marker written last, covering every artifact.
    files = {}
    for path in (pairs_path, votes_path, summary_path):
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
