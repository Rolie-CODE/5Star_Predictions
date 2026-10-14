# 5Star Predictions

Basketball prediction API built with FastAPI + XGBoost. Provides Home/Away winner predictions, Full-Time and Half-Time Over/Under predictions with trust percentages, and top daily picks.

---

## Quick Start

### 1. Clone & install
```bash
git clone <your-repo>
cd 5Star_Predictions
pip install -r requirements.txt
```

### 2. Set up environment
```bash
cp .env.example .env
# Edit .env and add your API_SPORTS_KEY
```

### 3. Train the ML models (first time only)
```bash
python -m app.ml.train
```
This generates `app/ml/models/winner_model.json`, `ft_ou_model.json`, `ht_ou_model.json`.

### 4. Run the server
```bash
uvicorn app.main:app --reload
```
API docs available at http://localhost:8000/docs

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/games/today` | All games today with predictions |
| `GET` | `/games/upcoming` | Games in the next 24 hours |
| `GET` | `/games/{game_id}` | Single game + full prediction detail |
| `GET` | `/predictions/top` | Top highest-trust picks of the day |
| `POST` | `/predictions/refresh` | Manually trigger prediction refresh |

### Query params for `/predictions/top`
- `type` — `winner` (default), `ft_ou`, `ht_ou`
- `limit` — number of picks (default 5, max 20)

---

## Scheduled Jobs

| Job | Schedule | What it does |
|-----|----------|--------------|
| `fetch_and_predict` | Daily 07:00 | Syncs today's games + odds, runs predictions |
| `fetch_results` | Every 2 hours | Updates finished game scores |
| `retrain_model` | 1st of month, 03:00 | Retrains XGBoost models with latest data |

---

## Deploy to Render

1. Push repo to GitHub
2. Connect repo in Render dashboard
3. Render auto-detects `render.yaml` — click **Apply**
4. Set `API_SPORTS_KEY` in Render → Environment Variables
5. Done ✅

---

## Switching DB (SQLite → PostgreSQL)

The only change needed is the `DATABASE_URL` env var:

```
# Local (SQLite)
DATABASE_URL=sqlite+aiosqlite:///./5star.db

# Render (PostgreSQL)
DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/dbname
```

`render.yaml` injects this automatically from the managed database.

---

## Project Structure

```
app/
├── main.py                  # FastAPI entry point + lifespan
├── core/
│   ├── config.py            # Settings (pydantic-settings)
│   └── scheduler.py         # APScheduler jobs
├── db/
│   ├── models.py            # SQLAlchemy ORM models
│   └── session.py           # Async engine + get_db dependency
├── ingestion/
│   ├── api_sports.py        # API-Sports HTTP client
│   └── data_pipeline.py     # Fetch → transform → persist
├── ml/
│   ├── features.py          # Feature engineering
│   ├── train.py             # Training script
│   ├── predict.py           # Inference engine
│   └── models/              # Saved .json model files (gitignored except .gitkeep)
├── services/
│   └── prediction_service.py
└── api/
    ├── schemas.py            # Pydantic response models
    └── routes/
        ├── games.py
        └── predictions.py
```
