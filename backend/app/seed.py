"""Creates tables (if missing) and seeds units/hospitals/a default dispatcher account."""
import random
from .database import Base, engine, SessionLocal
from .models import User, Unit, Hospital
from .auth import hash_password

random.seed(7)

# city grid roughly centered on an arbitrary metro area (lat/lon degrees)
CENTER_LAT, CENTER_LON = 30.75, 76.78  # Chandigarh, for flavor — purely illustrative

UNIT_NAMES = [f"Ambulance-{i+1:02d}" for i in range(10)]
HOSPITAL_NAMES = [
    "City General Hospital", "Northside Medical Center", "Riverside Hospital",
    "St. Mary's Trauma Center", "Eastgate Community Hospital", "Downtown Civil Hospital",
]


def _rand_point():
    return (CENTER_LAT + random.uniform(-0.08, 0.08), CENTER_LON + random.uniform(-0.08, 0.08))


def seed():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        if db.query(User).count() == 0:
            db.add(User(username="dispatcher", hashed_password=hash_password("dispatch123"), role="dispatcher"))
            db.add(User(username="admin", hashed_password=hash_password("admin123"), role="admin"))

        if db.query(Unit).count() == 0:
            for name in UNIT_NAMES:
                lat, lon = _rand_point()
                db.add(Unit(name=name, lat=lat, lon=lon, status="available",
                             active_load=0, speed_kmh=random.uniform(35, 55)))

        if db.query(Hospital).count() == 0:
            for name in HOSPITAL_NAMES:
                lat, lon = _rand_point()
                db.add(Hospital(name=name, lat=lat, lon=lon, capacity_pct=round(random.uniform(0.4, 0.9), 2)))

        db.commit()
    finally:
        db.close()


if __name__ == "__main__":
    seed()
    print("Database seeded.")
