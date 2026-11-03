import datetime as dt
from sqlalchemy import (Column, Integer, String, Float, DateTime, ForeignKey, Boolean)
from sqlalchemy.orm import relationship
from .database import Base


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String(64), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(32), default="dispatcher")  # dispatcher | admin
    created_at = Column(DateTime, default=dt.datetime.utcnow)


class Unit(Base):
    __tablename__ = "units"
    id = Column(Integer, primary_key=True)
    name = Column(String(64), nullable=False)
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)
    status = Column(String(32), default="available")  # available | dispatched | at_hospital
    active_load = Column(Integer, default=0)  # # of jobs handled in rolling window
    speed_kmh = Column(Float, default=45.0)

    dispatches = relationship("DispatchLog", back_populates="unit")


class Hospital(Base):
    __tablename__ = "hospitals"
    id = Column(Integer, primary_key=True)
    name = Column(String(64), nullable=False)
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)
    capacity_pct = Column(Float, default=0.7)  # fraction of beds FREE (1.0 = empty, 0 = full)


class Incident(Base):
    __tablename__ = "incidents"
    id = Column(Integer, primary_key=True)
    transcript = Column(String(1000), nullable=False)
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)
    severity_label = Column(String(16), nullable=True)       # critical | moderate | low
    severity_score = Column(Float, nullable=True)             # 0..1 model confidence-weighted
    status = Column(String(32), default="pending")            # pending | dispatched | resolved
    assigned_unit_id = Column(Integer, ForeignKey("units.id"), nullable=True)
    assigned_hospital_id = Column(Integer, ForeignKey("hospitals.id"), nullable=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow)
    dispatched_at = Column(DateTime, nullable=True)
    resolved_at = Column(DateTime, nullable=True)

    unit = relationship("Unit")
    hospital = relationship("Hospital")


class DispatchLog(Base):
    __tablename__ = "dispatch_log"
    id = Column(Integer, primary_key=True)
    incident_id = Column(Integer, ForeignKey("incidents.id"))
    unit_id = Column(Integer, ForeignKey("units.id"))
    hospital_id = Column(Integer, ForeignKey("hospitals.id"), nullable=True)
    score = Column(Float)
    eta_minutes = Column(Float)
    created_at = Column(DateTime, default=dt.datetime.utcnow)

    unit = relationship("Unit", back_populates="dispatches")
