import datetime as dt
from typing import Optional
from pydantic import BaseModel


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    role: str


class LoginRequest(BaseModel):
    username: str
    password: str


class IncidentCreate(BaseModel):
    transcript: str
    lat: float
    lon: float


class IncidentOut(BaseModel):
    id: int
    transcript: str
    lat: float
    lon: float
    severity_label: Optional[str]
    severity_score: Optional[float]
    status: str
    assigned_unit_id: Optional[int]
    assigned_hospital_id: Optional[int]
    created_at: dt.datetime

    class Config:
        from_attributes = True


class UnitOut(BaseModel):
    id: int
    name: str
    lat: float
    lon: float
    status: str
    active_load: int

    class Config:
        from_attributes = True


class HospitalOut(BaseModel):
    id: int
    name: str
    lat: float
    lon: float
    capacity_pct: float

    class Config:
        from_attributes = True
