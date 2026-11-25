"""Tests for app.votes_stance and its routes (synthetic, isolated).

See tests/conftest.py for the synthetic export, the temp SQLite database and
the in-memory loaded-model frame. The real evaluations.db is never touched.
"""
import pandas as pd
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


def test_overview_rows_shared_default(export_dir):
    # Default universe is "shared": only contribution 10 (a vote + a directional
    # comment). Contribution 20 has no directional comment, so it is excluded.
    rows = votes_stance.overview(source="gold")
    assert [r["contribution_id"] for r in rows] == [10]
    row10 = rows[0]
    assert row10["thumbs_up_share"] == 0.7
    assert row10["comment_favor_share"] == 0.5
    assert row10["counts"] == {"favor": 1, "against": 1, "neutral": 0, "mixed": 0}


def test_overview_rows_universe_all(export_dir):
    # universe="all" keeps every contribution, including non-shared cid 20.
    rows = votes_stance.overview(source="gold", universe="all")
    assert [r["contribution_id"] for r in rows] == [10, 20]


def test_overview_bad_universe_raises(export_dir):
    with pytest.raises(ValueError):
        votes_stance.overview(source="gold", universe="bogus")


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
    # Default universe is "shared": only the one shared contribution (cid 10).
    resp = client.get("/api/v1/votes-stance/overview?source=gold")
    assert resp.status_code == 200
    rows = resp.json()
    assert [r["contribution_id"] for r in rows] == [10]
    # universe=all returns every contribution.
    resp_all = client.get("/api/v1/votes-stance/overview?source=gold&universe=all")
    assert resp_all.status_code == 200
    assert len(resp_all.json()) == 2
    # An unknown universe is rejected by the Literal query type with 422.
    resp_bad = client.get("/api/v1/votes-stance/overview?source=gold&universe=bogus")
    assert resp_bad.status_code == 422


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
    # cid 20 is not shared, so use universe=all to keep it in the rows.
    resp = client.get("/api/v1/votes-stance/overview?source=live&universe=all")
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


# --------------------------------------------------------------------------
# Shared set, equivalence verdict and headline (gold only)
# --------------------------------------------------------------------------


def test_shared_contribution_ids(export_dir):
    # Only contribution 10 is shared: it has a thumbs vote (7/3) and a
    # directional comment (favor + against). Contribution 20 has a vote but only
    # neutral/mixed comments, so it never qualifies.
    export = votes_stance._load_export()
    assert votes_stance.shared_contribution_ids(export) == {"10"}


@pytest.mark.parametrize(
    "low, high, expected_verdict, expected_inside",
    [
        # Wholly inside the band -> equivalent (and inside_90 True).
        (-5.0, 5.0, "equivalent", True),
        # On both band edges -> still inside (bounds are inclusive).
        (-10.0, 10.0, "equivalent", True),
        # Straddling an edge -> undecided.
        (-11.0, 3.0, "undecided", False),
        (-3.0, 11.0, "undecided", False),
        # Wholly outside on one side -> not_equivalent.
        (11.0, 15.0, "not_equivalent", False),
        (-15.0, -11.0, "not_equivalent", False),
        # Just past an edge -> not_equivalent (strict boundary).
        (10.5, 12.0, "not_equivalent", False),
        # Missing bounds -> no verdict.
        (None, 5.0, None, False),
        (-5.0, None, None, False),
    ],
)
def test_equivalence_verdict(low, high, expected_verdict, expected_inside):
    result = votes_stance._equivalence_verdict(low, high, 10.0)
    assert result["margin_pp"] == 10.0
    assert result["verdict"] == expected_verdict
    assert result["inside_90"] is expected_inside


def test_headline_summary_block(export_dir):
    # Shared set is {10}: favor 1 / against 1, votes 7/3. The pooled gap and its
    # interval come verbatim from the synthetic shared-contributions block.
    headline = votes_stance.headline_summary_block()
    assert headline["pre_registered"] is False
    primary = headline["primary"]
    assert (primary["favor"], primary["against"]) == (1, 1)
    assert (primary["neutral"], primary["mixed"]) == (0, 0)
    assert (primary["vote_pro"], primary["vote_contra"]) == (7, 3)
    assert primary["vote_share"] == 70.0
    assert primary["favor_share"] == 50.0
    assert primary["difference_pp"] == -5.0
    assert primary["ci90"] == {"low": -10.0, "high": 1.0}
    assert headline["unweighted"] == {
        "difference_pp": -4.0,
        "ci90": {"low": -9.0, "high": 2.0},
    }
    assert headline["n_comments"] == 2
    assert headline["no_clear_stance"] == {"total": 0, "neutral": 0, "mixed": 0}
    assert headline["n_contributions"] == 1
    # ci90 [-10, 1] with margin 10: lower bound sits on the band edge, upper bound
    # inside -> wholly inside -> equivalent.
    assert headline["equivalence"]["verdict"] == "equivalent"
    assert headline["equivalence"]["inside_90"] is True
    assert headline["equivalence"]["margin_pp"] == 10.0
    assert headline["bootstrap"] == {"n": 10, "seed": 1}


def test_headline_summary_block_none_for_legacy_export(legacy_export_dir):
    # An older export without the shared-contributions block has no headline.
    assert votes_stance.headline_summary_block() is None


def test_shared_pooled_counts_empty_shared_set_gives_none_shares():
    # A contribution with only neutral/mixed comments has no directional comment,
    # so it is never in the shared set: both denominators are 0 -> None shares.
    export = {
        "comments": pd.DataFrame(
            {
                "contribution_id": ["1", "1"],
                "gold_category": ["neutral", "mixed"],
            }
        ),
        "votes": pd.DataFrame(
            {"contribution_id": ["1"], "votingPro": [5], "votingContra": [5]}
        ),
    }
    counts = votes_stance._shared_pooled_counts(export)
    assert counts["n_contributions"] == 0
    assert counts["vote_pro"] == 0
    assert counts["vote_contra"] == 0
    assert counts["favor"] == 0
    assert counts["against"] == 0
    assert counts["vote_share"] is None
    assert counts["favor_share"] is None


def test_shared_pooled_counts_sums_neutral_and_mixed_over_shared():
    # One shared contribution carrying every category: neutral and mixed are
    # summed over the shared set (they are not in the favour share, but counted).
    export = {
        "comments": pd.DataFrame(
            {
                "contribution_id": ["1"] * 5,
                "gold_category": [
                    "in_favor",
                    "in_favor",
                    "against",
                    "neutral",
                    "mixed",
                ],
            }
        ),
        "votes": pd.DataFrame(
            {"contribution_id": ["1"], "votingPro": [8], "votingContra": [2]}
        ),
    }
    counts = votes_stance._shared_pooled_counts(export)
    assert counts["n_contributions"] == 1
    assert counts["favor"] == 2
    assert counts["against"] == 1
    assert counts["neutral"] == 1
    assert counts["mixed"] == 1
    # favor_share excludes neutral/mixed: 2 / (2 + 1) -> 66.67%.
    assert counts["favor_share"] == pytest.approx(100.0 * 2 / 3)
    assert counts["vote_share"] == 80.0


# --------------------------------------------------------------------------
# Per-key-message agreement (gold only)
# --------------------------------------------------------------------------


def test_contribution_key_messages_counts_and_comments(export_dir):
    rows = votes_stance.contribution_key_messages(10)
    # Two key messages, in first-appearance order.
    assert [r["key_message"] for r in rows] == ["km-10-a", "km-10-b"]
    km_a = rows[0]
    assert (km_a["favor"], km_a["against"], km_a["neutral"]) == (3, 1, 0)
    assert km_a["n_with_stance"] == 4
    assert km_a["too_few"] is False
    # The comments behind each count are listed, in order.
    assert [c["comment_text"] for c in km_a["comments"]["favor"]] == [
        "kmA comment 1",
        "kmA comment 2",
        "kmA comment 3",
    ]
    assert [c["comment_text"] for c in km_a["comments"]["against"]] == ["kmA comment 4"]
    assert km_a["comments"]["neutral"] == []


def test_contribution_key_messages_too_few(export_dir):
    rows = votes_stance.contribution_key_messages(20)
    assert len(rows) == 1
    km = rows[0]
    assert (km["favor"], km["against"], km["neutral"]) == (0, 1, 1)
    assert km["n_with_stance"] == 1
    assert km["too_few"] is True


def test_contribution_key_messages_unknown_raises(export_dir):
    with pytest.raises(KeyError):
        votes_stance.contribution_key_messages(999)


def test_contribution_key_messages_empty_for_legacy_export(legacy_export_dir):
    # No artifact -> empty list, never an error.
    assert votes_stance.contribution_key_messages(10) == []


# --------------------------------------------------------------------------
# Highlight stance + spans
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "favor, against, expected",
    [
        (3, 1, "favor"),      # more in favour
        (1, 3, "against"),    # more against
        (2, 2, "disputed"),   # evenly split, non-zero
        (0, 0, "neutral"),    # no clear stance either way
    ],
)
def test_highlight_stance(favor, against, expected):
    assert votes_stance._highlight_stance(favor, against) == expected


def test_contribution_key_messages_attach_spans(export_dir):
    rows = votes_stance.contribution_key_messages(10)
    km_a, km_b = rows[0], rows[1]
    # km-10-a has two located passages; spans are plain-int start/end dicts.
    assert km_a["spans"] == [{"start": 0, "end": 10}, {"start": 20, "end": 30}]
    assert km_b["spans"] == [{"start": 5, "end": 15}]
    # 3 favor vs 1 against -> favor highlight; provenance names the model.
    assert km_a["highlight_stance"] == "favor"
    assert km_a["spans_source"] == "extraction_model:mistral-nemo"


def test_contribution_key_messages_no_located_passage(export_dir):
    # km-20-a has an against-leaning gold stance but no located passage.
    rows = votes_stance.contribution_key_messages(20)
    km = rows[0]
    assert km["spans"] == []
    assert km["highlight_stance"] == "against"
    assert km["spans_source"] == "extraction_model:mistral-nemo"


def test_contribution_key_messages_spans_source_none_without_artifact(tmp_path, monkeypatch):
    # An export WITH key messages but WITHOUT the span artifact: spans are empty
    # and spans_source is None (no highlights to attribute).
    from tests.conftest import _write_export

    directory = tmp_path / "votes_stance_nospans"
    _write_export(directory, include_spans=False)
    monkeypatch.setattr(votes_stance, "VOTE_STANCE_EXPORT_DIR", directory)
    votes_stance._load_export.cache_clear()
    try:
        rows = votes_stance.contribution_key_messages(10)
        assert all(r["spans"] == [] for r in rows)
        assert all(r["spans_source"] is None for r in rows)
        # The colour category is still derived from the gold counts.
        assert rows[0]["highlight_stance"] == "favor"
    finally:
        votes_stance._load_export.cache_clear()


def test_route_contribution_key_messages_spans(export_dir, client):
    resp = client.get("/api/v1/contributions/10/votes-stance/key-messages")
    assert resp.status_code == 200
    body = resp.json()
    assert body[0]["spans"] == [{"start": 0, "end": 10}, {"start": 20, "end": 30}]
    assert body[0]["highlight_stance"] == "favor"
    assert body[0]["spans_source"] == "extraction_model:mistral-nemo"


# --------------------------------------------------------------------------
# Key-message overview (gold only)
# --------------------------------------------------------------------------


def test_key_message_overview_ranks_and_excludes_thin(export_dir):
    overview = votes_stance.key_message_overview()
    agreed = overview["most_agreed"]
    rejected = overview["most_rejected"]
    # km-20-a is too thin (1 stance comment) and appears in neither list.
    assert all(i["key_message"] != "km-20-a" for i in agreed + rejected)
    # Most agreed: km-10-a (3 in favor) before km-10-b (2 in favor).
    assert [i["key_message"] for i in agreed] == ["km-10-a", "km-10-b"]
    # Most rejected: both have 1 against, tiebreak on against share -> km-10-b
    # (1/3) ranks above km-10-a (1/4).
    assert [i["key_message"] for i in rejected] == ["km-10-b", "km-10-a"]
    assert all(not i["too_few"] for i in agreed + rejected)


def test_key_message_overview_limit(export_dir):
    overview = votes_stance.key_message_overview(limit=1)
    assert len(overview["most_agreed"]) == 1
    assert overview["most_agreed"][0]["key_message"] == "km-10-a"


def test_key_message_overview_empty_for_legacy_export(legacy_export_dir):
    overview = votes_stance.key_message_overview()
    assert overview == {"most_agreed": [], "most_rejected": []}


def test_key_message_overview_restricts_to_shared_and_threshold(tmp_path, monkeypatch):
    # Extra key messages: km-20-b would rank first (3 in favour) but lives on the
    # non-shared contribution 20, and km-10-c has only 2 stance comments. The
    # overview must drop the first by the shared restriction and the second by
    # the stance-count threshold, leaving only the shared, rankable key messages.
    from tests.conftest import _write_export

    directory = tmp_path / "votes_stance_kmrestrict"
    extra = [
        (20, "km-20-b", "kmY comment 1", "in_favor"),
        (20, "km-20-b", "kmY comment 2", "in_favor"),
        (20, "km-20-b", "kmY comment 3", "in_favor"),
        (10, "km-10-c", "kmC comment 1", "in_favor"),
        (10, "km-10-c", "kmC comment 2", "in_favor"),
    ]
    _write_export(directory, extra_key_message_rows=extra)
    monkeypatch.setattr(votes_stance, "VOTE_STANCE_EXPORT_DIR", directory)
    votes_stance._load_export.cache_clear()
    try:
        overview = votes_stance.key_message_overview()
        names = [i["key_message"] for i in overview["most_agreed"]]
        # km-20-b is excluded because contribution 20 is not shared.
        assert "km-20-b" not in names
        # km-10-c is excluded because it has fewer than three stance comments.
        assert "km-10-c" not in names
        # The shared, rankable key messages remain, most agreed first.
        assert names == ["km-10-a", "km-10-b"]
    finally:
        votes_stance._load_export.cache_clear()


# --------------------------------------------------------------------------
# Routes (gold only; no source/model/evaluator params)
# --------------------------------------------------------------------------


def test_route_votes_stance_summary(export_dir, client):
    resp = client.get("/api/v1/votes-stance/summary")
    assert resp.status_code == 200
    body = resp.json()
    assert body["snapshot_utc"] == "20260102T030405060708Z"
    # The response carries only the headline block now; no overall/shared keys.
    assert set(body) == {"snapshot_utc", "headline"}
    headline = body["headline"]
    assert headline["pre_registered"] is False
    primary = headline["primary"]
    assert (primary["vote_pro"], primary["vote_contra"]) == (7, 3)
    assert (primary["favor"], primary["against"]) == (1, 1)
    assert primary["vote_share"] == 70.0
    assert primary["favor_share"] == 50.0
    assert primary["difference_pp"] == -5.0
    assert primary["ci90"] == {"low": -10.0, "high": 1.0}
    assert headline["n_contributions"] == 1
    assert headline["n_comments"] == 2
    assert headline["no_clear_stance"] == {"total": 0, "neutral": 0, "mixed": 0}
    assert headline["unweighted"]["difference_pp"] == -4.0
    assert headline["equivalence"]["verdict"] == "equivalent"
    assert headline["bootstrap"] == {"n": 10, "seed": 1}


def test_route_votes_stance_summary_headline_null_for_legacy(legacy_export_dir, client):
    resp = client.get("/api/v1/votes-stance/summary")
    assert resp.status_code == 200
    body = resp.json()
    assert body["headline"] is None


def test_route_contribution_key_messages(export_dir, client):
    resp = client.get("/api/v1/contributions/10/votes-stance/key-messages")
    assert resp.status_code == 200
    body = resp.json()
    assert [r["key_message"] for r in body] == ["km-10-a", "km-10-b"]
    assert body[0]["comments"]["favor"][0]["comment_text"] == "kmA comment 1"


def test_route_contribution_key_messages_unknown_404(export_dir, client):
    resp = client.get("/api/v1/contributions/999/votes-stance/key-messages")
    assert resp.status_code == 404


def test_route_contribution_key_messages_empty_for_legacy(legacy_export_dir, client):
    resp = client.get("/api/v1/contributions/10/votes-stance/key-messages")
    assert resp.status_code == 200
    assert resp.json() == []


def test_route_key_message_overview(export_dir, client):
    resp = client.get("/api/v1/votes-stance/key-messages/overview")
    assert resp.status_code == 200
    body = resp.json()
    assert [i["key_message"] for i in body["most_agreed"]] == ["km-10-a", "km-10-b"]


def test_route_key_message_overview_limit_bounds(export_dir, client):
    # Out-of-range limit is rejected by the Query bounds (ge=1, le=100).
    assert client.get(
        "/api/v1/votes-stance/key-messages/overview?limit=0"
    ).status_code == 422
    assert client.get(
        "/api/v1/votes-stance/key-messages/overview?limit=101"
    ).status_code == 422
    assert client.get(
        "/api/v1/votes-stance/key-messages/overview?limit=1"
    ).status_code == 200
