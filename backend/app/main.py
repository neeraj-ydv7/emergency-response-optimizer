import datetime as dt
import json
import os
from typing import List

from fastapi import FastAPI, Depends, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from jose import jwt, JWTError
from sqlalchemy.orm import Session

from .database import get_db, engine, Base
from .models import User, Unit, Hospital, Incident, DispatchLog
from .schemas import LoginRequest, Token, IncidentCreate, IncidentOut, UnitOut, HospitalOut
from .auth import authenticate_user, create_access_token, get_current_user, SECRET_KEY, ALGORITHM
from .seed import seed
from . import ml, dispatch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.normpath(os.path.join(HERE, "..", "frontend"))

app = FastAPI(title="Emergency Response Optimizer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # same-origin app; loosened only because this is a demo/prototype
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    Base.metadata.create_all(bind=engine)
    seed()
    ml.get_metrics()  # warms the model cache, fails loudly at boot if models are missing


# ---------------------------------------------------------------- WebSocket hub
class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, message: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_text(json.dumps(message, default=str))
            except Exception:
                dead.append(ws)
        for d in dead:
            self.disconnect(d)


manager = ConnectionManager()


def _ws_user_ok(token: str, db: Session):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        return db.query(User).filter(User.username == username).first() is not None
    except JWTError:
        return False


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket, token: str = Query(...)):
    db = next(get_db())
    if not _ws_user_ok(token, db):
        await websocket.close(code=4401)
        return
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()  # client doesn't need to send anything; keep-alive
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# ---------------------------------------------------------------- Auth
@app.post("/api/auth/login", response_model=Token)
def login(body: LoginRequest, db: Session = Depends(get_db)):
    user = authenticate_user(db, body.username, body.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    token = create_access_token({"sub": user.username, "role": user.role})
    return Token(access_token=token, username=user.username, role=user.role)


# ---------------------------------------------------------------- Reference data
@app.get("/api/units", response_model=List[UnitOut])
def list_units(db: Session = Depends(get_db), _user: User = Depends(get_current_user)):
    return db.query(Unit).all()


@app.get("/api/hospitals", response_model=List[HospitalOut])
def list_hospitals(db: Session = Depends(get_db), _user: User = Depends(get_current_user)):
    return db.query(Hospital).all()


@app.get("/api/incidents", response_model=List[IncidentOut])
def list_incidents(db: Session = Depends(get_db), _user: User = Depends(get_current_user)):
    return db.query(Incident).order_by(Incident.created_at.desc()).limit(200).all()


@app.get("/api/model_metrics")
def model_metrics(_user: User = Depends(get_current_user)):
    return ml.get_metrics()


@app.get("/api/stats")
def stats(db: Session = Depends(get_db), _user: User = Depends(get_current_user)):
    total = db.query(Incident).count()
    resolved = db.query(Incident).filter(Incident.status == "resolved").count()
    pending = db.query(Incident).filter(Incident.status == "pending").count()
    dispatched = db.query(Incident).filter(Incident.status == "dispatched").count()
    by_sev = {}
    for label in ["critical", "moderate", "low"]:
        by_sev[label] = db.query(Incident).filter(Incident.severity_label == label).count()
    avg_score = db.query(DispatchLog).count()
    return {
        "total_incidents": total, "resolved": resolved, "pending": pending,
        "dispatched": dispatched, "by_severity": by_sev, "total_dispatches": avg_score,
    }


# ---------------------------------------------------------------- Core workflow: report -> triage -> dispatch
@app.post("/api/incidents", response_model=IncidentOut)
async def create_incident(body: IncidentCreate, db: Session = Depends(get_db)):
    """Public intake endpoint (a 911-style call-in is not behind a dispatcher
    login in real systems either) — triage + dispatch happen immediately and
    the result is pushed to every connected dispatcher dashboard over the
    WebSocket.
    """
    label, confidence, sev_score = ml.classify_severity(body.transcript)

    incident = Incident(
        transcript=body.transcript, lat=body.lat, lon=body.lon,
        severity_label=label, severity_score=sev_score, status="pending",
    )
    db.add(incident)
    db.commit()
    db.refresh(incident)

    _dispatch_incident(db, incident)

    await manager.broadcast({
        "type": "incident_update",
        "incident": IncidentOut.model_validate(incident).model_dump(),
    })
    return incident


def _dispatch_incident(db: Session, incident: Incident):
    available_units = db.query(Unit).filter(Unit.status == "available").all()
    if not available_units:
        return  # stays pending; no free units right now (genuine resource contention)

    ranked = dispatch.rank_units(incident.lat, incident.lon, available_units, incident.severity_score or 0.5)
    if not ranked:
        return
    best = ranked[0]
    unit = db.get(Unit, best.unit_id)

    hospitals = db.query(Hospital).all()
    hosp_choice = dispatch.choose_hospital(incident.lat, incident.lon, unit.lat, unit.lon, hospitals)

    unit.status = "dispatched"
    unit.active_load = min(unit.active_load + 1, 5)

    incident.status = "dispatched"
    incident.assigned_unit_id = unit.id
    incident.assigned_hospital_id = hosp_choice["hospital_id"] if hosp_choice else None
    incident.dispatched_at = dt.datetime.utcnow()

    log = DispatchLog(
        incident_id=incident.id, unit_id=unit.id,
        hospital_id=hosp_choice["hospital_id"] if hosp_choice else None,
        score=best.score, eta_minutes=best.eta_min,
    )
    db.add(log)
    db.commit()


@app.post("/api/incidents/{incident_id}/resolve", response_model=IncidentOut)
async def resolve_incident(incident_id: int, db: Session = Depends(get_db),
                            _user: User = Depends(get_current_user)):
    incident = db.get(Incident, incident_id)
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    incident.status = "resolved"
    incident.resolved_at = dt.datetime.utcnow()
    if incident.assigned_unit_id:
        unit = db.get(Unit, incident.assigned_unit_id)
        if unit:
            unit.status = "available"
            unit.active_load = max(unit.active_load - 1, 0)
    db.commit()
    db.refresh(incident)
    await manager.broadcast({
        "type": "incident_update",
        "incident": IncidentOut.model_validate(incident).model_dump(),
    })
    return incident


# ---------------------------------------------------------------- Frontend (static, no build step)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def serve_login():
    return FileResponse(os.path.join(FRONTEND_DIR, "login.html"))


@app.get("/dashboard")
def serve_dashboard():
    return FileResponse(os.path.join(FRONTEND_DIR, "dashboard.html"))
