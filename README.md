# 5Star Predictions

Basketball prediction API built with FastAPI + XGBoost. Provides Home/Away winner predictions and Full-Time Over/Under predictions with calibrated trust percentages, plus top daily picks.

---

## Quick Start

### 1. Clone & install
```bash
git clone <your-repo>
cd 5Star_Predictions
pip install -r requirements-dev.txt   # app + pytest; Python 3.14 (see .python-version)
```

### 2. Set up environment
```bash
cp .env.example .env
# Edit .env: add your API_SPORTS_KEY and an ADMIN_API_KEY
```

### 3. Models
Trained models are committed in `app/ml/models/`, so the server runs without
training anything. To retrain, you need the training data (see below):
```bash
python -m app.ml.train          # add --no-db to skip the app database
git add app/ml/models && git commit -m "Retrain models"
```

**Training data is not in git.** Put these files in `data/` locally (the folder
is gitignored):
- `data/kaggle_nba/nba_games_all.csv` and `data/kaggle_nba/nba_betting_totals.csv` —
  from [NBA Historical Stats and Betting Data](https://www.kaggle.com/datasets/ehallmar/nba-historical-stats-and-betting-data)
  (license listed as unknown, so it isn't redistributed here)
- `data/api_sports_nba_historical_2022_2025.csv` — exported from API-Sports with
  your own key
- Optional, same Kaggle dataset: `data/kaggle_nba/nba_betting_money_line.csv`
  (used by the backtest's sportsbook baseline)
- Optional: `data/kaggle_nba/csv/game.csv` from the
  [NBA Basketball Database](https://www.kaggle.com/datasets/wyattowalsh/basketball).
  It fills the seasons between the two sources above (2018-19 to 2021-22) and
  is used to match API-Sports teams to NBA.com ids.

Training also uses every finished game from past seasons that the app has
stored in its database (point `DATABASE_URL` at the production database to
include those). The current season is left out: the server replays it itself.

All games are replayed as one history to build each game's pre-game
features: Elo ratings that carry over between seasons, season-to-date net
rating and win rates, recent form, rest days and back-to-backs, and scoring
relative to the league average. Training saves the state after the last game
to `app/ml/models/carryover.json`; the server starts each season from it and
replays the season's stored games on top. Retrain after each season so the
carryover includes it.

Only regular-season games are used. Each model is fitted on the oldest 70% of
games, calibrated on the next 15%, and scored on the newest 15%; the results
are written to `app/ml/models/metadata.json`.

| Model | Test accuracy | Notes |
|-------|---------------|-------|
| Winner | ~65.0% | Trust is a calibrated probability. Backtest 2010–2024: 65.7%, against 65.6% for Elo alone and 68.7% for the sportsbook favourite |
| Full-time O/U | ~50.1% | **No measurable edge** over the sportsbook line; loses about 2% per bet in the backtest |
| Half-time O/U | — | Not served: no verified historical first-half lines |

Picks are not betting advice; at standard -110 prices you need ~52.4% just to
break even.

### 4. Run the server
```bash
uvicorn app.main:app --reload
```
Database migrations run automatically at startup. API docs available at
http://localhost:8000/docs

### 5. Run the tests
```bash
pytest
```

### 6. Backtest the models
```bash
python -m app.ml.backtest       # about a minute; needs the training data
```
Rebuilds the models once per season from 2010 on, using only earlier seasons,
and compares them with "always home" and a basic Elo rating. Over/Under picks
are scored by profit per bet at -110. Results go to
`app/ml/models/backtest.json`. Run it before and after any model change.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check (`ok` or `degraded`) |
| `GET` | `/games/today` | Today's slate (US Eastern day) with predictions |
| `GET` | `/games/upcoming` | Games in the next 24 hours |
| `GET` | `/games/{game_id}` | Single game + full prediction detail |
| `GET` | `/predictions/top` | Top highest-trust picks |
| `GET` | `/predictions/performance` | Stored picks graded against final scores (`season`, `model_version` optional) |
| `POST` | `/predictions/refresh` | Run the full pipeline now — requires `X-API-Key: <ADMIN_API_KEY>` |

`full_time_ou` is `null` for games with no sportsbook line yet, and
`half_time_ou` is always `null` until a validated half-time model exists.

### Query params for `/predictions/top`
- `type` — `winner` (default), `ft_ou`, `ht_ou`
- `limit` — number of picks (default `TOP_PREDICTIONS_LIMIT`, max 20)

---

## Data source and API-Sports plan limits

On the **Free** API-Sports plan:
- Current-season queries (`league` + `season`) are rejected, so games are
  fetched by date only (yesterday → tomorrow is allowed) and filtered to the NBA.
- Odds are available; lines are the median main line across all bookmakers.
- 100 requests/day and 10/minute. The client throttles itself to
  `API_SPORTS_MAX_REQUESTS_PER_MINUTE`; the scheduled jobs use about 65/day
  on a busy day.
- Past seasons are limited to 2022–2024.

Team stats are computed from the finished games stored in the app's own
database (same code as training). On the Free plan the database only knows the
games synced since the app started, so start it at the beginning of the season
and keep it running. A paid plan lets the app backfill the whole season in
one request at startup.

---

## Scheduled Jobs

Times are US Eastern (`LEAGUE_TIMEZONE`).

| Job | Schedule | What it does |
|-----|----------|--------------|
| `catch_up` | At startup | Syncs yesterday's results and predicts today's and tomorrow's games if any are missing. On the Free plan, results from before yesterday can't be fetched, so days the server was down for longer are lost |
| `fetch_and_predict` | Daily 10:00 and 17:00 | Syncs results, today's and tomorrow's games + lines, team stats; runs predictions |
| `fetch_results` | 19:00–03:00, every 2 hours | Updates scores (yesterday + today) and team stats |

Models are retrained locally, not on the server (see step 3).

---

## Deploy to Render

1. Push repo to GitHub
2. Connect repo in Render dashboard
3. Render auto-detects `render.yaml` — click **Apply**
4. Set `API_SPORTS_KEY` in Render → Environment Variables
   (`ADMIN_API_KEY` is generated for you; copy it from the dashboard)
5. Set `CORS_ORIGINS` to your frontend's origin once you have one

Free-plan notes:
- The web service sleeps after ~15 minutes without traffic, which pauses the
  scheduled jobs (`catch_up` runs when it wakes). To keep jobs on time, have a
  free uptime service (e.g. cron-job.org) request `/health` every 10 minutes;
  one always-on service fits in the free 750 hours/month.
- Render's free databases expire — check their current limits. On the Free
  API-Sports plan the database holds the only copy of this season's results,
  so consider a non-expiring free Postgres (e.g. Neon or Supabase) and set its
  URL as `DATABASE_URL` (see the comment in `render.yaml`).

---

## Database

`DATABASE_URL` selects the database:

```
# Local (SQLite)
DATABASE_URL=sqlite+aiosqlite:///./5star.db

# PostgreSQL — postgres:// and postgresql:// URLs are switched to asyncpg automatically
DATABASE_URL=postgresql://user:pass@host:5432/dbname
```

Schema changes are Alembic migrations in `migrations/versions/`. After
changing `app/db/models.py`:

```bash
alembic revision --autogenerate -m "describe the change"
```

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
│   └── session.py           # Async engine, get_db dependency, migrations
├── ingestion/
│   ├── api_sports.py        # API-Sports HTTP client (throttled)
│   └── data_pipeline.py     # Fetch → transform → persist, team stats
├── ml/
│   ├── features.py          # Feature engineering shared by training + inference
│   ├── train.py             # Training script
│   ├── predict.py           # Inference engine + calibration
│   └── models/              # Trained model files + metadata.json (committed)
├── services/
│   └── prediction_service.py
└── api/
    ├── schemas.py            # Pydantic response models
    └── routes/
        ├── games.py
        └── predictions.py
migrations/                  # Alembic migrations
tests/                       # pytest suite
data/                        # Training data (local only, gitignored)
```
