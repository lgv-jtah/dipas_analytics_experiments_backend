"""Tests for app.votes_stance and its routes (synthetic, isolated).

See tests/conftest.py for the synthetic export, the temp SQLite database and
the in-memory loaded-model frame. The real evaluations.db is never touched.
"""
import pytest

from app import votes_stance
from tests.conftest import add_stance_evaluation


# --------------------------------------------------------------------------
# Pure roll-up rules (the rule table the frontend and test must pin)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "labels,expected",
    [
        ({"in_favor"}, "in_favor"),
        ({"against"}, "against"),
        ({"neutral"}, "neutral"),
        ({"in_favor", "against"}, "mixed"),
        ({"in_favor", "neutral"}, "in_favor"),
        ({"against", "neutral"}, "against"),
        ({"in_favor", "against", "neutral"}, "mixed"),
        (set(), "neutral"),
    ],
)
def test_category_from_labels_rule_table(labels, expected):
    assert votes_stance.category_from_labels(labels) == expected


def test_favor_share_excludes_mixed():
    counts = votes_stance.favor_counts(["in_favor", "against", "mixed", "neutral"])
    assert counts["favor"] == 1
    assert counts["against"] == 1
    assert counts["mixed"] == 1
    assert counts["neutral"] == 1
    # mixed excluded: favor / (favor + against) = 1 / 2
    assert counts["favor_share"] == 0.5


def test_favor_share_none_when_no_directional():
    counts = votes_stance.favor_counts(["neutral", "mixed"])
    assert counts["favor_share"] is None


# --------------------------------------------------------------------------
# Loading and provenance
# --------------------------------------------------------------------------


def test_available_models(export_dir):
    assert votes_stance.available_models() == ["base", "gptoss-120b"]


def test_snapshot_info(export_dir):
    info = votes_stance.snapshot_info()
    assert info["snapshot_utc"] == "20260102T030405060708Z"
    assert info["n_contributions"] == 2
    assert info["default_model"] == "gptoss-120b"
    assert info["note"] == votes_stance.DESCRIPTIVE_ONLY_NOTE


def test_load_export_detects_tampering(export_dir):
    # Corrupt the comments file after the hashes marker is written.
    (export_dir / votes_stance.COMMENTS_FILE).write_bytes(b"corrupted")
    votes_stance._load_export.cache_clear()
    with pytest.raises(RuntimeError):
        votes_stance.available_models()


# --------------------------------------------------------------------------
# Gold / model panels
# --------------------------------------------------------------------------


def test_contribution_panel_gold_counts(export_dir):
    panel = votes_stance.contribution_panel(10, source="gold")
    assert panel["comments"]["favor"] == 1
    assert panel["comments"]["against"] == 1
    assert panel["comments"]["comment_favor_share"] == 0.5
    assert panel["votes"]["thumbs_up_share"] == 0.7
    assert panel["votes"]["votes_pro"] == 7


def test_contribution_panel_gold_mixed_excluded(export_dir):
    # Contribution 20: comment C mixed, comment D neutral -> no directional.
    panel = votes_stance.contribution_panel(20, source="gold")
    assert panel["comments"]["mixed"] == 1
    assert panel["comments"]["neutral"] == 1
    assert panel["comments"]["comment_favor_share"] is None


def test_contribution_panel_model_differs_from_gold(export_dir):
    # base rolls comment C to in_favor (not mixed like gold) and flips comment
    # D stays neutral -> contribution 20 has one favor comment under base.
    panel = votes_stance.contribution_panel(20, source="model", model="base")
    assert panel["comments"]["favor"] == 1
    assert panel["comments"]["mixed"] == 0
    assert panel["comments"]["model"] == "base"


def test_overview_rows(export_dir):
    rows = votes_stance.overview(source="gold")
    assert [r["contribution_id"] for r in rows] == [10, 20]
    row10 = rows[0]
    assert row10["thumbs_up_share"] == 0.7
    assert row10["comment_favor_share"] == 0.5
    assert row10["counts"] == {"favor": 1, "against": 1, "neutral": 0, "mixed": 0}


# --------------------------------------------------------------------------
# Validation and lookup errors
# --------------------------------------------------------------------------


def test_unknown_source_raises(export_dir):
    with pytest.raises(ValueError):
        votes_stance.contribution_panel(10, source="bogus")


def test_unknown_model_raises(export_dir):
    with pytest.raises(ValueError):
        votes_stance.contribution_panel(10, source="model", model="no-such-model")


def test_unknown_contribution_raises(export_dir):
    with pytest.raises(KeyError):
        votes_stance.contribution_panel(999, source="gold")


# --------------------------------------------------------------------------
# Live resolution
# --------------------------------------------------------------------------


def test_live_correct_resolves_to_loaded_model(export_dir, loaded_model_df, db_session):
    # A "correct" verdict on comment A resolves to the loaded model's label.
    add_stance_evaluation(
        db_session, contribution_id=10, key_message="kmA",
        comment_text="comment A", verdict="correct", evaluator="alice",
    )
    cats = votes_stance.live_comment_categories(db_session)
    assert cats[votes_stance._comment_id(10, "comment A")] == "in_favor"


def test_live_incorrect_uses_corrected_label(export_dir, loaded_model_df, db_session):
    add_stance_evaluation(
        db_session, contribution_id=10, key_message="kmA", comment_text="comment A",
        verdict="incorrect", corrected_label="ablehnung", evaluator="alice",
    )
    cats = votes_stance.live_comment_categories(db_session)
    assert cats[votes_stance._comment_id(10, "comment A")] == "against"


def test_live_incorrect_without_corrected_label_dropped(export_dir, loaded_model_df, db_session):
    add_stance_evaluation(
        db_session, contribution_id=10, key_message="kmA", comment_text="comment A",
        verdict="incorrect", corrected_label=None, evaluator="alice",
    )
    cats = votes_stance.live_comment_categories(db_session)
    assert cats == {}


def test_live_mixed_comment_rolls_up(export_dir, loaded_model_df, db_session):
    # Comment C has two pairs; correct on kmC1 (in favor) + corrected to
    # ablehnung on kmC2 -> the comment rolls up to mixed.
    add_stance_evaluation(
        db_session, contribution_id=20, key_message="kmC1", comment_text="comment C",
        verdict="correct", evaluator="alice",
    )
    add_stance_evaluation(
        db_session, contribution_id=20, key_message="kmC2", comment_text="comment C",
        verdict="incorrect", corrected_label="ablehnung", evaluator="alice",
    )
    cats = votes_stance.live_comment_categories(db_session)
    assert cats[votes_stance._comment_id(20, "comment C")] == "mixed"


def test_live_evaluator_filter(export_dir, loaded_model_df, db_session):
    add_stance_evaluation(
        db_session, contribution_id=10, key_message="kmA", comment_text="comment A",
        verdict="correct", evaluator="alice",
    )
    add_stance_evaluation(
        db_session, contribution_id=10, key_message="kmB", comment_text="comment B",
        verdict="correct", evaluator="bob",
    )
    alice = votes_stance.live_comment_categories(db_session, evaluator="alice")
    assert len(alice) == 1
    everyone = votes_stance.live_comment_categories(db_session)
    assert len(everyone) == 2


def test_live_panel_empty_when_no_data(export_dir, loaded_model_df, db_session):
    # Live with no verdicts: empty counts, not an error.
    panel = votes_stance.contribution_panel(10, source="live", db=db_session)
    assert panel["comments"]["n_comments"] == 0
    assert panel["comments"]["comment_favor_share"] is None
    # Coverage is present and honest: nothing verified, denominator still real.
    assert panel["comments"]["live_pairs_verified"] == 0
    assert panel["comments"]["live_pairs_total"] == 2  # contribution 10: A (1) + B (1)
    assert panel["comments"]["live_comments_partially_verified"] == 0


def test_live_partial_verification_is_explicit(export_dir, loaded_model_df, db_session):
    # Comment C has two loaded-model pairs (kmC1, kmC2); verify only kmC1.
    add_stance_evaluation(
        db_session, contribution_id=20, key_message="kmC1", comment_text="comment C",
        verdict="correct", evaluator="alice",
    )
    panel = votes_stance.contribution_panel(20, source="live", db=db_session)
    comments = panel["comments"]
    # Category comes from the single verified pair (in_favor), counted once.
    assert comments["favor"] == 1
    assert comments["n_comments"] == 1
    # Coverage makes the partial verification explicit, not silent.
    assert comments["live_pairs_verified"] == 1
    assert comments["live_pairs_total"] == 3  # C(2) + D(1)
    assert comments["live_comments_partially_verified"] == 1
    assert comments["live_comments_fully_verified"] == 0


def test_live_full_verification_counts_as_fully(export_dir, loaded_model_df, db_session):
    # Verify both of comment C's pairs -> fully verified.
    add_stance_evaluation(
        db_session, contribution_id=20, key_message="kmC1", comment_text="comment C",
        verdict="correct", evaluator="alice",
    )
    add_stance_evaluation(
        db_session, contribution_id=20, key_message="kmC2", comment_text="comment C",
        verdict="correct", evaluator="alice",
    )
    panel = votes_stance.contribution_panel(20, source="live", db=db_session)
    assert panel["comments"]["live_comments_fully_verified"] == 1
    assert panel["comments"]["live_comments_partially_verified"] == 0
    assert panel["comments"]["live_pairs_verified"] == 2


def test_non_live_sources_have_no_coverage(export_dir):
    panel = votes_stance.contribution_panel(10, source="gold")
    assert panel["comments"]["live_pairs_verified"] is None
    assert panel["comments"]["live_pairs_total"] is None


# --------------------------------------------------------------------------
# HTTP routes
# --------------------------------------------------------------------------


def test_route_snapshot_info(export_dir, client, db_session):
    add_stance_evaluation(
        db_session, contribution_id=10, key_message="kmA", comment_text="comment A",
        verdict="correct", evaluator="alice",
    )
    resp = client.get("/api/v1/votes-stance/snapshot-info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["default_source"] == "gold"
    assert body["evaluators"] == ["alice"]
    assert body["available_models"] == ["base", "gptoss-120b"]


def test_route_contribution_panel(export_dir, client):
    resp = client.get("/api/v1/contributions/10/votes-stance?source=gold")
    assert resp.status_code == 200
    body = resp.json()
    assert body["contribution_id"] == 10
    assert body["comments"]["comment_favor_share"] == 0.5


def test_route_overview(export_dir, client):
    resp = client.get("/api/v1/votes-stance/overview?source=gold")
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 2


def test_route_bad_source_422(export_dir, client):
    # The Literal query type rejects an unknown source with 422.
    resp = client.get("/api/v1/contributions/10/votes-stance?source=bogus")
    assert resp.status_code == 422


def test_route_bad_model_400(export_dir, client):
    resp = client.get(
        "/api/v1/contributions/10/votes-stance?source=model&model=no-such-model"
    )
    assert resp.status_code == 400


def test_route_unknown_contribution_404(export_dir, client):
    resp = client.get("/api/v1/contributions/999/votes-stance?source=gold")
    assert resp.status_code == 404


def test_route_live_empty_ok(export_dir, loaded_model_df, client):
    resp = client.get("/api/v1/contributions/10/votes-stance?source=live")
    assert resp.status_code == 200
    assert resp.json()["comments"]["n_comments"] == 0


def test_route_overview_live_coverage(export_dir, loaded_model_df, client, db_session):
    add_stance_evaluation(
        db_session, contribution_id=20, key_message="kmC1", comment_text="comment C",
        verdict="correct", evaluator="alice",
    )
    resp = client.get("/api/v1/votes-stance/overview?source=live")
    assert resp.status_code == 200
    row20 = next(r for r in resp.json() if r["contribution_id"] == 20)
    assert row20["live_pairs_verified"] == 1
    assert row20["live_pairs_total"] == 3
    assert row20["live_comments_partially_verified"] == 1


def test_route_unknown_evaluator_400(export_dir, client, db_session):
    add_stance_evaluation(
        db_session, contribution_id=10, key_message="kmA", comment_text="comment A",
        verdict="correct", evaluator="alice",
    )
    # A known evaluator passes; an unknown one is rejected.
    assert client.get(
        "/api/v1/votes-stance/overview?source=gold&evaluator=alice"
    ).status_code == 200
    bad = client.get("/api/v1/votes-stance/overview?source=gold&evaluator=zzz")
    assert bad.status_code == 400
    bad_panel = client.get(
        "/api/v1/contributions/10/votes-stance?source=live&evaluator=zzz"
    )
    assert bad_panel.status_code == 400
