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
| `GET` | `/contributions/{id}/key-messages/{key_message}/stances` | Predicted stances for a (contribution, key message) pair |

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
