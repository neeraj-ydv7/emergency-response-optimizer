# Emergency Response Optimizer — Deployable Prototype

A working full-stack application implementing the core dispatch system described
in the research paper *"Emergency Response Optimizer: An Intelligent System for
Real-Time Emergency Management."* This is the deployable counterpart to the
earlier offline analysis prototype — it has a real backend, a real database, a
real authenticated frontend, and live (not simulated-offline) dispatch decisions.

## What's actually real here

- **FastAPI backend** with a REST API and a WebSocket channel — not a script.
- **SQLite database (via SQLAlchemy)** persisting users, ambulance units,
  hospitals, incidents, and a full dispatch log — not in-memory lists.
- **JWT-based authentication** (bcrypt-hashed passwords) protecting the
  dispatcher dashboard and every data-returning endpoint. The public incident
  report endpoint is intentionally open, mirroring how a real 911 call-in isn't
  behind a dispatcher's own login either.
- **A trained severity-triage model** (TF-IDF + XGBoost) that classifies each
  incoming transcript as critical / moderate / low in real time.
- **Algorithm 1's multi-objective dispatch scoring**, exactly as formalized in
  Section III-A of the paper, running live against the current state of all
  units and hospitals in the database — not pre-computed.
- **A live dispatcher dashboard** (Leaflet map, incident feed, stats panel)
  that updates over a WebSocket the instant a new incident is triaged and
  dispatched, with no page refresh.

## What's a disclosed, drop-in-replaceable substitution

Consistent with the paper's honesty about sandbox constraints:

| Paper component | This prototype uses | Why |
|---|---|---|
| DistilBERT transformer triage | TF-IDF + XGBoost | No internet access to huggingface.co in the development sandbox |
| OSRM + live traffic API routing | Haversine distance + per-unit average speed | No internet access to routing/map servers in the development sandbox |

Both substitutions are isolated to `app/ml.py` and `app/dispatch.py` respectively
— swapping in the real services later does not require touching the database
schema, the API, or the frontend.

## What this prototype does NOT implement (honest scope)

Per the paper's own framing, four Tier-2 items are **proposed extensions /
future work**, not implemented here: multimodal (text+audio) triage fusion,
predictive ambulance pre-positioning, a generative-AI dispatcher copilot, and
federated learning across agencies. Nothing in this codebase claims otherwise.

## Project structure

```
ero_app/
├── run.sh                      # one-command install + train + launch
├── backend/
│   ├── requirements.txt
│   ├── train_model.py          # generates synthetic transcripts, trains triage model
│   ├── models/                 # trained model artifacts (created by train_model.py)
│   ├── data/                   # ero.db (SQLite, created on first run)
│   └── app/
│       ├── main.py             # FastAPI app: REST + WebSocket + static file serving
│       ├── database.py         # SQLAlchemy engine/session
│       ├── models.py           # User, Unit, Hospital, Incident, DispatchLog tables
│       ├── schemas.py          # Pydantic request/response models
│       ├── auth.py             # bcrypt hashing + JWT issue/verify
│       ├── seed.py             # seeds demo units/hospitals/accounts on first boot
│       ├── ml.py                # severity triage inference
│       └── dispatch.py         # Algorithm 1 multi-objective scoring
└── frontend/
    ├── login.html              # JWT login
    └── dashboard.html          # Leaflet map + live feed + stats (vanilla JS, no build step)
```

## Running it

```bash
./run.sh
```

This installs dependencies, trains the triage model on first run (takes under a
minute), seeds the database with 10 ambulances and 6 hospitals, and starts the
server at **http://localhost:8000**.

Demo accounts:
- `dispatcher` / `dispatch123`
- `admin` / `admin123`

Open the URL, log in, and use the "Report an Incident" form to submit a
transcript (try something like *"my husband collapsed and isn't breathing"*
for a critical case, or *"has a small cut that needs a bandage"* for a low
one). The dashboard shows the triage result, the dispatched unit, the chosen
hospital, and updates live.

## Security notes (scoped to what this topic needs)

- Passwords are never stored in plaintext — bcrypt hashing only.
- Dispatcher-facing data (units, hospitals, incident list, stats, model
  metrics) requires a valid JWT; tokens expire after 8 hours (one shift).
- The WebSocket connection itself is authenticated with the same JWT.
- `ERO_SECRET_KEY` should be set via environment variable before any real
  deployment; a development default is used otherwise and the code says so.
- This is a course/conference prototype, not a production emergency system:
  there's no rate limiting, HTTPS termination, or audit-grade logging, which
  would be required before handling real 911 traffic.

## Reproducing the triage model's metrics

```bash
cd backend && python3 train_model.py
```

Prints precision/recall/F1/accuracy for both the XGBoost model and the naive
keyword-heuristic baseline on a held-out test split — the same metrics citable
in the paper's Section IV/V.
