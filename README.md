# DIPAS Analytics Evaluation API

FastAPI backend for human evaluation of model-predicted **key messages** and **stances** in participatory planning contributions.

## Setup

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

The server starts at `http://localhost:8000`.  
Interactive API docs are at `http://localhost:8000/docs`.

By default the API reads `whole_process_contributions_with_comments_250_deepsseek-v4-pro_deepsseek-v4-pro.parquet` from the project root. To use a different dataset, set the `PARQUET_PATH` environment variable to the path of another parquet file:

```bash
PARQUET_PATH=./my_other_dataset.parquet uvicorn app.main:app --reload
```

---

## Project structure

```
.
├── app/
│   ├── main.py       # FastAPI app, CORS, router registration
│   ├── routes.py     # All API endpoints
│   ├── schemas.py    # Pydantic request/response models
│   ├── models.py     # SQLAlchemy ORM models (SQLite)
│   ├── database.py   # DB engine & session factory
│   └── data.py       # Parquet loader & typed accessors
├── whole_process_contributions_with_comments_250_deepsseek-v4-pro_deepsseek-v4-pro.parquet  # default dataset; override via PARQUET_PATH
├── requirements.txt
└── evaluations.db    # auto-created on first run
```

---

## API overview

All endpoints are prefixed with `/api/v1`.

### Contributions

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/contributions` | List all 120 contributions |
| `GET` | `/contributions/{id}` | Get one contribution |

### Key messages

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/contributions/{id}/key-messages` | Predicted key messages for a contribution |

Each key message includes `key_message_type` (`Zustand` / `Wunsch` / `Problem` / `Qualität`) and `key_message_explanation`.

### Stances

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/contributions/{id}/stances?key_message=...` | Predicted stances for a (contribution, key message) pair |

Each stance entry includes the `comment_text`, `stance` (`in favor` / `neutral` / `ablehnung`), and the model's `explanation`.

### Evaluations

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/evaluations/key-messages` | Submit a key-message verdict |
| `GET`  | `/evaluations/key-messages` | List key-message evaluations (filterable) |
| `POST` | `/evaluations/stances` | Submit a stance verdict |
| `GET`  | `/evaluations/stances` | List stance evaluations (filterable) |
| `GET`  | `/evaluations/stats` | Overall evaluation progress |

#### POST `/evaluations/key-messages`

```json
{
  "contribution_id": 10813,
  "key_message": "Bogenstraße ist in der Rush Hour voll mit Kindern auf Fahrrädern",
  "verdict": "correct",
  "comment": "optional free-text note",
  "evaluator": "user-id-or-name"
}
```

#### POST `/evaluations/stances`

```json
{
  "contribution_id": 10813,
  "key_message": "Bogenstraße ist in der Rush Hour voll mit Kindern auf Fahrrädern",
  "comment_text": "Das ist richtig. Radstreifen zu schmal ...",
  "verdict": "incorrect",
  "comment": "optional note",
  "evaluator": "user-id-or-name"
}
```

`verdict` must be `"correct"` or `"incorrect"`.  
Re-submitting the same `(contribution_id, key_message[, comment_text], evaluator)` tuple **updates** the existing record.

#### GET `/evaluations/stats`

```json
{
  "total_key_messages": 397,
  "evaluated_key_messages": 12,
  "correct_key_messages": 9,
  "incorrect_key_messages": 3,
  "total_stances": 938,
  "evaluated_stances": 45,
  "correct_stances": 38,
  "incorrect_stances": 7
}
```

---

## Votes vs. comment stance (descriptive only)

A descriptive view that compares, per contribution, the thumbs-up share among
votes with the favor share among comment stances. These per-contribution
figures are **descriptive only** and are **not** part of the pre-registered
vote-ratio equivalence test.

### Endpoints

All prefixed with `/api/v1`.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/votes-stance/snapshot-info` | Frozen snapshot time, available models, default source/model and the known evaluator list |
| `GET` | `/votes-stance/overview?universe=` | One descriptive row per contribution (vote share, favor share, counts) for the scatter and table; `universe` is `shared` (default) or `all` |
| `GET` | `/contributions/{id}/votes-stance` | Vote counts and rolled-up comment stance for one contribution |
| `GET` | `/votes-stance/summary` | Votes-vs-stance headline (gold only), computed over the shared contributions |
| `GET` | `/contributions/{id}/votes-stance/key-messages` | Per-key-message gold agreement/disagreement for one contribution |
| `GET` | `/votes-stance/key-messages/overview?limit=` | Most agreed-with and most rejected gold key messages (`limit` 1–100, default 20) |

`source` selects the comment labels: `gold` (frozen gold labels), `model` (a
model's predictions; the `model` query param is then required) or `live`
(resolved from stored human verdicts against the loaded model). For `live`,
an optional `evaluator` query param restricts to one annotator. An unknown
`model` or `evaluator` returns 400; an unknown contribution returns 404.

`/votes-stance/overview` also takes `universe`: `shared` (the default) restricts
the rows to the shared contributions (those with both a thumbs vote and a
comment carrying a clear stance — 80 on the `20261105` export), while `all` keeps
every contribution (115). `all` exists for debugging; the UI always uses the
default `shared` set. An unknown `universe` returns 400 (422 at the route, which
validates it against the allowed literal).

#### Gold-only summary and key-message endpoints

The last three endpoints are **gold (hand-label) only**: they take no `source`,
`model` or `evaluator` parameters and always read the frozen gold labels. The UI
likewise exposes no source/model selectors for these views.

- `GET /votes-stance/summary` returns `{snapshot_utc, headline}`. The whole
  votes-vs-stance view runs on the **shared contributions** — those with both at
  least one thumbs vote and at least one gold comment carrying a clear stance (on
  the `20261105` export, 80 contributions). `headline` is the pooled
  thumbs-vs-comment-stance picture over that set: the thumbs and comment-stance
  counts, the favour and thumbs-up shares, the pooled gap and its 90% interval, a
  secondary unweighted gap, and an `equivalence` verdict. The verdict reads the
  pooled 90% interval against a ±10 pp band: `equivalent` when the interval lies
  wholly inside the band, `not_equivalent` when it lies wholly outside on one
  side, `undecided` otherwise. `headline` is `null` for an older export that
  predates the shared-contributions block.
- `GET /contributions/{id}/votes-stance/key-messages` lists one entry per gold
  key message of the contribution, in first-appearance order, with in-favour /
  against / neutral comment counts and the comment texts behind each count. An
  unknown contribution is a 404; when the key-message artifact is absent (an
  older export) the list is empty. Each entry also carries the fields that drive
  the contribution-page highlights:
  - `spans` — a list of `{start, end}` half-open character offsets into the
    contribution text, marking that key message's located source passage(s).
    Empty when none could be located; empty for every key message when the span
    artifact is absent.
  - `highlight_stance` — the highlight **colour category** for those spans,
    derived from the gold comment counts: `favor` when more comments favour the
    key message, `against` when more oppose it, `disputed` when the two
    directional counts are equal and non-zero, and `neutral` when no comment
    takes a clear stance.
  - `spans_source` — provenance for the offsets, e.g.
    `extraction_model:mistral-nemo`, or `null` when no span artifact is present.
    It makes explicit that the highlighted positions come from the extraction
    model, **not** from the hand labels.

  **Span offsets index into the extraction model's text.** `spans` are character
  offsets into the `mistral-nemo` `contributionContent` (that is the text the
  export measured against), so `PARQUET_PATH` **must** stay pinned to the
  `mistral-nemo` parquet. The backend serves that same `contribution_content`;
  pointing `PARQUET_PATH` at a different model would shift the displayed text and
  make every highlight point at the wrong passage.
- `GET /votes-stance/key-messages/overview` returns `most_agreed` and
  `most_rejected` lists. Only key messages of the shared contributions are
  ranked, so the overview runs on the same contribution set as the rest of the
  view.

**Thin-data rule.** A key message is included in the overview only when it has at
least **three** comments carrying a clear stance (in favour or against);
below that, counts are too thin to rank and the key message is flagged `too_few`
and dropped from the lists.

**Ranking rule.** The overview reports **counts only, never percentages**, and
never groups key messages across contributions (each (contribution, key message)
pair is its own item). `most_agreed` is ordered by in-favour comment count
descending, `most_rejected` by against count descending; ties break by
directional share, then by contribution id and key message for a fully
deterministic order.

### The data export

The view is backed by a frozen export produced offline by the analysis repo,
not by any live computation here. It is read from the directory named by the
`VOTE_STANCE_EXPORT_DIR` environment variable, defaulting to
`data_exports/votes_stance/`. The directory holds these files, all
hash-verified against the completion marker on load:

- the per-comment table (`votes_stance_comments.parquet`),
- the per-key-message table (`votes_stance_key_messages.parquet`) — one row per
  gold scoring pair at the key-message grain, backing the per-key-message
  endpoints; optional, so an older export without it degrades those views to
  empty rather than failing,
- the per-key-message highlight spans (`votes_stance_key_message_spans.parquet`)
  — one row per gold key message giving its located source spans over the
  `mistral-nemo` contribution text (and the `extraction_model` they index into);
  it backs the `spans` / `highlight_stance` / `spans_source` fields above. Also
  optional: an export produced before highlights existed simply has no artifact,
  `spans_source` is then `null`, and the contribution page shows plain text,
- the vote snapshot (`votes_snapshot.parquet`),
- the summary (`votes_stance_export.summary.json`), and
- the completion marker (`artifact_hashes.json`).

If the directory or marker is missing the view is simply unavailable (the
frontend hides the panel).

The summary's **`shared_contributions`** block backs `GET /votes-stance/summary`.
It restricts both sides to the shared contributions — those having both a thumbs
vote and a clear-stance comment — and carries the pooled and unweighted gaps,
each with its 90% interval, plus the bootstrap parameters (`n`, `seed`) those
figures were resampled with. From it the backend builds the `headline` response
(`app.votes_stance.headline_summary_block`): it reads the pooled/unweighted gaps
and intervals straight from the block, derives the pooled thumbs
(`vote_pro`/`vote_contra`) and comment-stance (`favor`/`against`/`neutral`/`mixed`)
counts over the shared set from the export parquets
(`app.votes_stance._shared_pooled_counts`), and computes the `equivalence`
verdict by reading the pooled 90% interval against the ±10 pp band
(`EQUIVALENCE_MARGIN_PP`). On the `20261105` export (`VOTE_STANCE_EXPORT_DIR` in
`docker-compose`) the shared set is the 80 contributions that have both thumbs
and a clear comment: 654/93 thumbs, 129/27 directional comments, plus 44 neutral
and 8 mixed comments that are counted but excluded from `favor_share`. See
`stance-detection-app/docs/votes_stance_key_messages_note.md` for the wider
analysis context.

The block is optional; an older export without it makes `headline` `null`.

To refresh the export, run the exporter in the `stance-detection-app` repo and
copy its files into this backend:

```bash
# in stance-detection-app/
uv run python cli.py votes-stance-export
# then copy ALL files from the new
# dataset/experiments/votes_stance_export_<timestamp>/ directory into
# this backend's data_exports/votes_stance_<timestamp>/
```

### Deploying a refreshed export

The export directory is mounted into the backend container and selected with
`VOTE_STANCE_EXPORT_DIR`. To deploy a new export:

1. Copy the new export into a fresh, timestamped directory under
   `data_exports/`, e.g. `data_exports/votes_stance_<timestamp>/` (keep the old
   one so you can roll back).
2. Point the compose mount and `VOTE_STANCE_EXPORT_DIR` at the new directory in
   `docker-compose.yml`.
3. Rebuild and restart (user step):

   ```bash
   docker compose up -d --build
   ```

### The `live` source rule

A `live` comment category is resolved per pair from the stored human verdicts,
then rolled up per comment:

- `correct` → the label the annotator reviewed, i.e. the **loaded model's**
  label for that pair.
- `incorrect` with a `corrected_label` → the **corrected** label.
- `incorrect` without a `corrected_label` → **dropped** (unresolved).

Only verified pairs contribute, so a comment may be rolled up from a subset of
its pairs. Such partial coverage is reported explicitly (per-contribution
pairs verified / total and fully- vs. partially-verified comment counts), never
presented as a full category.

### Running the tests

```bash
uv run --no-project --with-requirements requirements.txt --with pytest --with pytest-cov --with httpx pytest -q
```
