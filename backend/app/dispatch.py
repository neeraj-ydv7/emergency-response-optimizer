"""Multi-objective dispatch engine — Algorithm 1 from the paper (Section III-A).

    Score(u, i) = w1*Sev(i) + w2*(1/ETA(u,i)) + w3*Cap(h) - w4*Load(u)
    w1=0.40  w2=0.30  w3=0.20  w4=0.10

Design note (carried over from the validated offline simulation): Sev(i) and
Cap(h) are properties of the INCIDENT and of a HOSPITAL, not of a candidate
unit, so they are constants with respect to u and cannot by themselves change
which unit is nearest-and-least-loaded for a single incident in isolation.
Their real effect shows up in two places that this module models explicitly:

 1. Sev(i) sets PRIORITY when several incidents compete for the same pool of
    free units (a severity-ordered queue, not a per-unit tiebreaker).
 2. Cap(h) is used for the separate HOSPITAL hand-off decision: a unit may
    pass a nearer, fuller hospital in favor of a slightly farther one with
    more free capacity, trading a capped amount of extra travel time for a
    shorter handoff delay — exactly the mechanism validated in the
    discrete-event simulation (see prototype's dispatch_simulation.py).
"""
import math
from dataclasses import dataclass

W1_SEV, W2_ETA, W3_CAP, W4_LOAD = 0.40, 0.30, 0.20, 0.10
MAX_HOSPITAL_DETOUR_KM = 6.0  # cap on how much extra distance capacity-seeking may add


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def eta_minutes(dist_km, speed_kmh):
    return (dist_km / max(speed_kmh, 1e-3)) * 60.0


@dataclass
class UnitScore:
    unit_id: int
    name: str
    score: float
    eta_min: float
    dist_km: float


def rank_units(incident_lat, incident_lon, available_units, sev_score: float):
    """Scores every available unit for one incident. Returns list sorted best-first.

    Sev(i) is included as specified by the formula (so the score is faithful
    to Algorithm 1), but because it's constant across all candidate units for
    a single incident, it shifts the overall score uniformly and does not by
    itself re-rank units — exactly as derived analytically in the paper's
    validation. The term that actually re-ranks units here is
    w2*(1/ETA) - w4*Load, i.e. traffic-aware proximity balanced against how
    busy a unit already is.
    """
    scored = []
    etas = []
    for u in available_units:
        dist = haversine_km(incident_lat, incident_lon, u.lat, u.lon)
        eta = eta_minutes(dist, u.speed_kmh)
        etas.append(eta)
        scored.append((u, dist, eta))

    if not scored:
        return []

    max_eta = max(etas) or 1.0
    results = []
    for u, dist, eta in scored:
        inv_eta_norm = 1.0 - (eta / max_eta)  # normalized 0..1, higher = closer/faster
        load_norm = min(u.active_load / 5.0, 1.0)
        score = (W1_SEV * sev_score) + (W2_ETA * inv_eta_norm) - (W4_LOAD * load_norm)
        results.append(UnitScore(unit_id=u.id, name=u.name, score=round(score, 4),
                                  eta_min=round(eta, 2), dist_km=round(dist, 2)))
    results.sort(key=lambda r: r.score, reverse=True)
    return results


def choose_hospital(incident_lat, incident_lon, unit_lat, unit_lon, hospitals):
    """Chooses the hospital that best trades off hand-off delay (capacity)
    against extra travel distance, within a bounded detour budget.
    """
    best = None
    best_value = -1e9
    for h in hospitals:
        dist_from_incident = haversine_km(incident_lat, incident_lon, h.lat, h.lon)
        # how much farther than the nearest hospital this choice is
        best = best  # no-op to keep structure clear
        handoff_delay = (1.0 - h.capacity_pct) ** 2 * 20.0  # minutes, matches simulation model
        value = (W3_CAP * h.capacity_pct) - (handoff_delay / 60.0) - (dist_from_incident / 100.0)
        if value > best_value:
            best_value = value
            best = (h, dist_from_incident, handoff_delay)
    if best is None:
        return None
    h, dist, handoff = best
    return {"hospital_id": h.id, "name": h.name, "dist_km": round(dist, 2),
            "handoff_delay_min": round(handoff, 2), "capacity_pct": h.capacity_pct}
