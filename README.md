# AI Query Optimizer

A research project that uses **PPO reinforcement learning** to choose the optimal
**join order** for SQL queries on PostgreSQL, protects the database from
**Denial-of-Service (DoS) attacks** with an **Isolation Forest** anomaly detector,
and estimates the **carbon footprint** of every query it runs.

It ships as a three-tier system wired together on a single host:

```
React (Vite, :5173)  →  Spring Boot (Java, :8080)  →  FastAPI (Python, :8000)  →  PostgreSQL (:5432)
```

The browser talks **only** to Spring Boot. FastAPI is the AI engine and is never
exposed to the frontend; Spring Boot proxies the analytics endpoints to it.

---

## Three pillars

| Pillar | Component | What it does |
|---|---|---|
| **Join-order optimization** | `ai_engine/agent/ppo_agent.py`, `ai_engine/vectorizer/schema_vectorizer.py` | A 3-layer attention network samples a join permutation, which is emitted as a `pg_hint_plan` `Leading(...)` hint. Trained with clipped-PPO2 batches and a REINFORCE fallback. |
| **DoS detection** | `ai_engine/anomaly/isolation_forest.py`, `middleware/.../security/AstFirewall.java` | An sklearn Isolation Forest learns the normal `(wait_time_ms, active_connections)` relationship and flags the "high wait + low connections" victim pattern. An AST firewall blocks non-SELECT SQL and obvious tautologies. |
| **Carbon tracking** | `middleware/.../service/CarbonEstimator.java` | Converts exec time + disk-block reads into energy (kWh) and CO₂ (grams) per query, with session totals. |

---

## Repository layout

```
ai query optimizer/
├── middleware/                 # Spring Boot 4 (Java 25) — gateway, telemetry, carbon, firewall
│   ├── pom.xml                 # uses the Maven wrapper (mvnw.cmd)
│   └── src/main/java/com/aioptimizer/middleware/
│       ├── controller/         # QueryController, AnalyticsController, CarbonController
│       ├── service/            # QueryTelemetryService, QueryFeatureExtractor, CarbonEstimator
│       ├── security/           # AstFirewall
│       ├── config/             # CarbonProperties
│       └── model/              # CarbonResult
├── ai_engine/                  # FastAPI (Python 3.13) — RL agent, anomaly detector, analytics
│   ├── main.py                 # app entrypoint; mounts all routers
│   ├── routers/                # queryOptimizer, feedback, analytics_router, anomaly_router
│   ├── agent/                  # PPO model + trainer, experience buffer, singleton
│   ├── vectorizer/             # table stats → 11-dim feature tensor
│   ├── anomaly/                # Isolation Forest wrapper + singleton
│   ├── analytics/              # in-memory query history store
│   ├── models/                 # Pydantic request/response models
│   ├── run_queries.py          # fires ~60 benchmark queries
│   └── requirements.txt
├── frontend/                   # React 18 + Vite + Recharts
│   ├── src/App.jsx             # 4 tabs: Query Runner, Dashboard, History, Security
│   └── src/api/api.js          # all calls to /api (proxied to :8080)
├── Dockerfile                  # PostgreSQL 18 + pg_hint_plan
└── README.md

imdb_data/                      # IMDb relational dataset (21 CSV tables + schematext.sql)
```

---

## Prerequisites

| Tool | Version | Notes |
|---|---|---|
| JDK | **25** | `pom.xml` targets Java 25; required by Spring Boot 4.1 |
| Maven | 3.9+ | not required globally — the wrapper `mvnw.cmd` downloads it |
| Python | 3.13 | for the FastAPI engine |
| Node.js + npm | 18+ | for the Vite frontend |
| PostgreSQL | **18** | with the `pg_hint_plan` extension (see `Dockerfile`) |

---

## Setup

### 1. Database — PostgreSQL 18 + pg_hint_plan

Build/run the bundled image (installs `pg_hint_plan` and preloads it):

```bash
docker build -t imdb-pg18 .
docker run -d --name imdb-pg -p 5432:5432 \
  -e POSTGRES_PASSWORD=password -e POSTGRES_DB=ai_optimizer imdb-pg18
```

Then load the IMDb schema and data:

```bash
psql -h localhost -U postgres -d ai_optimizer -f imdb_data/schematext.sql
# load each CSV into its matching table, e.g.:
#   \copy title FROM 'imdb_data/title.csv' (FORMAT csv)
```

> The column order in each CSV matches the `CREATE TABLE` order in
> [`imdb_data/schematext.sql`](<imdb_data/schematext.sql>). Add `HEADER` only if
> your CSV files include a header row.

### 2. AI Engine — FastAPI (:8000)

```bash
cd "ai query optimizer/ai_engine"
python -m venv .venv && .\.venv\Scripts\activate        # optional
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Health check: `GET http://localhost:8000/` → `{"message": "server is running successfully."}`

### 3. Middleware — Spring Boot (:8080)

```bash
cd "ai query optimizer/middleware"
.\mvnw.cmd spring-boot:run
```

(If Maven is installed globally, `mvn spring-boot:run` also works.)

### 4. Frontend — Vite (:5173)

```bash
cd "ai query optimizer/frontend"
npm install
npm run dev
```

Open http://localhost:5173. The Vite dev server proxies `/api` → `http://localhost:8080`.

---

## Configuration

### Database (used by both Spring Boot and FastAPI)

| Env var | Spring Boot key | Default |
|---|---|---|
| `DB_URL` | `spring.datasource.url` | `jdbc:postgresql://localhost:5432/ai_optimizer` |
| `DB_USERNAME` | `spring.datasource.username` | `postgres` |
| `DB_PASSWORD` | `spring.datasource.password` | `password` |
| `DB_HOST` | (FastAPI vectorizer) | `localhost` |
| `DB_PORT` | (FastAPI vectorizer) | `5432` |
| `DB_NAME` | (FastAPI vectorizer) | `ai_optimizer` |

`DB_HOST` / `DB_PORT` / `DB_NAME` / `DB_USERNAME` / `DB_PASSWORD` are read by
`ai_engine/vectorizer/schema_vectorizer.py`; the Spring side reads `DB_URL`,
`DB_USERNAME`, `DB_PASSWORD` from `application.properties` (which are themselves
`${ENV:default}` placeholders).

### Carbon model — `application.properties`

| Key | Default | Meaning |
|---|---|---|
| `carbon.tdp-watts` | `150` | server CPU thermal design power (W) |
| `carbon.intensity-g-per-kwh` | `708.0` | grid CO₂ intensity (g/kWh, India CEA 2023) |
| `carbon.ssd-watt-per-gb` | `2.0` | SSD read power (W/GB) |

---

## How a query flows

1. **Frontend** → `POST /api/query/execute` to Spring Boot.
2. **AstFirewall** parses the SQL and rejects non-SELECT / tautologies.
3. **QueryFeatureExtractor** (JSQLParser AST walk) collects `(alias, realName)`
   pairs and join edges.
4. **Spring Boot** → `POST /optimize` to FastAPI with tables, real names, join
   edges, and an `aliases_safe` security flag.
5. **FastAPI** builds an 11-dim feature tensor from live `pg_catalog` stats,
   samples a join order via the attention policy, and returns an
   `/*+ Leading(...) */`-hinted SQL plus XAI attention weights.
6. **Spring Boot** executes the hinted SQL under a `statement_timeout` guard and
   measures `wait_time_ms` / `exec_time_ms` / `latency_ms` + active connections.
7. **CarbonEstimator** computes CPU + I/O energy and CO₂.
8. **Spring Boot** → `POST /feedback` with telemetry; FastAPI scores the anomaly
   detector, computes the reward, and accumulates a PPO batch (fires at size 8).
9. Response (optimized SQL, timing, carbon, anomaly status) returns to the UI.

---

## API reference

### Spring Boot (:8080)

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/query/execute` | run + optimize a query |
| GET | `/api/analytics/summary` | aggregate stats |
| GET | `/api/analytics/history` | paginated query log |
| GET | `/api/analytics/improvement?n=` | reward trend |
| GET | `/api/analytics/rewards` | raw reward list |
| GET | `/api/analytics/ppo/stats` | PPO training state |
| GET | `/api/analytics/anomaly` | Isolation Forest status |
| GET | `/api/carbon/summary` | session carbon totals |

### FastAPI (:8000) — not browser-facing

| Method | Path | Purpose |
|---|---|---|
| POST | `/optimize` | sample join order + build hint |
| POST | `/feedback` | reward + anomaly + PPO batch |
| GET | `/feedback/stats` | training/buffer/checkpoint state |
| GET | `/feedback/buffer` | raw replay buffer (debug) |
| GET | `/analytics/summary` | aggregate stats |
| GET | `/analytics/history` | paginated query log |
| GET | `/analytics/improvement` | reward trend |
| GET | `/analytics/rewards` | raw reward list |
| GET | `/anomaly/status` | Isolation Forest config/state |

---

## Running the benchmark

With all three services up, fire the built-in workload:

```bash
cd "ai query optimizer/ai_engine"
python run_queries.py
```

It sends ~60 IMDb multi-table join queries through the full pipeline, then points
you at `GET http://localhost:8000/feedback/stats` to inspect learning progress.

---

## Security notes

- `log_prob_old` is stored **server-side** in the experience buffer and is never
  returned to (or accepted from) the client — it cannot be tampered with.
- Table aliases are whitelisted (`^[A-Za-z_][A-Za-z0-9_]*$`) in both Java and
  Python; an unsafe alias suppresses the hint entirely rather than risking
  comment-injection SQL injection.
- Anomalous samples are **quarantined** — never fed back into the Isolation
  Forest — so the detector cannot learn attacks as "normal".
