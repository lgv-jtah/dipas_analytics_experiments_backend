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
| `GET` | `/votes-stance/overview` | One descriptive row per contribution (vote share, favor share, counts) for the scatter and table |
| `GET` | `/contributions/{id}/votes-stance` | Vote counts and rolled-up comment stance for one contribution |
| `GET` | `/votes-stance/summary` | Overall summary (gold only): the pre-registered headline plus the exploratory shared-contributions block |
| `GET` | `/contributions/{id}/votes-stance/key-messages` | Per-key-message gold agreement/disagreement for one contribution |
| `GET` | `/votes-stance/key-messages/overview?limit=` | Most agreed-with and most rejected gold key messages (`limit` 1–100, default 20) |

`source` selects the comment labels: `gold` (frozen gold labels), `model` (a
model's predictions; the `model` query param is then required) or `live`
(resolved from stored human verdicts against the loaded model). For `live`,
an optional `evaluator` query param restricts to one annotator. An unknown
`model` or `evaluator` returns 400; an unknown contribution returns 404.

#### Gold-only summary and key-message endpoints

The last three endpoints are **gold (hand-label) only**: they take no `source`,
`model` or `evaluator` parameters and always read the frozen gold labels. The UI
likewise exposes no source/model selectors for these views.

- `GET /votes-stance/summary` returns `overall` (the **pre-registered** headline,
  copied verbatim from the frozen export — favor vs. thumbs-up share, the point
  difference and its 90% interval) and `shared` (the **not pre-registered**
  shared-contributions block, which carries its own `pre_registered: false` flag
  and a note). Either may be `null` for an older export that predates these
  blocks.
- `GET /contributions/{id}/votes-stance/key-messages` lists one entry per gold
  key message of the contribution, in first-appearance order, with in-favour /
  against / neutral comment counts and the comment texts behind each count. An
  unknown contribution is a 404; when the key-message artifact is absent (an
  older export) the list is empty.
- `GET /votes-stance/key-messages/overview` returns `most_agreed` and
  `most_rejected` lists.

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
- the vote snapshot (`votes_snapshot.parquet`),
- the summary (`votes_stance_export.summary.json`), and
- the completion marker (`artifact_hashes.json`).

If the directory or marker is missing the view is simply unavailable (the
frontend hides the panel).

The summary carries two blocks that back `GET /votes-stance/summary`:

- **`overall_summary`** — the **pre-registered** headline figures, copied
  verbatim from the frozen vote-ratio analysis summary and never recomputed
  (`pre_registered: true`).
- **`shared_contributions`** — an **exploratory, not pre-registered** comparison
  that restricts both sides to the contributions having both a thumbs vote and a
  clear-stance comment (`pre_registered: false`, with its own note). See
  `stance-detection-app/docs/votes_stance_key_messages_note.md` for why this does
  not replace the pre-registered result.

Both blocks are optional; an older export without them makes the respective
field `null`.

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
