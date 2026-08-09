"""
StateSnapshot builder — hydrates protobuf from TimescaleDB.

Queries the canonical numerical state at a given simulation time
and assembles the StateSnapshot message for agent deliberation.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from google.protobuf.timestamp_pb2 import Timestamp

from engine.proto.v1 import simulation_pb2 as pb
from engine.store.postgres import PostgresStore


async def build_snapshot(
    store: PostgresStore,
    sim_time: datetime,
    patient_id: int = 1,
) -> pb.StateSnapshot:
    """
    Build a StateSnapshot from the database at the given simulation time.

    Queries:
      - Latest vital signs before sim_time
      - Active drug infusions at sim_time
      - Pending agent messages
    """
    ts = Timestamp()
    ts.FromDatetime(sim_time)

    snapshot = pb.StateSnapshot(
        version_hash=_hash_snapshot(sim_time, patient_id),
        sim_time=ts,
        sdi_score=0.0,
        shared_effect_compartment=0.0,
    )

    if store is None:
        return snapshot

    # ------------------------------------------------------------------
    # Vital signs — latest value per metric before sim_time
    # ------------------------------------------------------------------
    vital_rows = await store.fetch(
        """
        SELECT DISTINCT ON (metric_code)
               metric_code, value, source, time
        FROM vital_signs
        WHERE patient_id = $1 AND time <= $2
        ORDER BY metric_code, time DESC
        """,
        patient_id,
        sim_time,
    )
    for row in vital_rows:
        unit = _metric_unit(row["metric_code"])
        vt = Timestamp()
        vt.FromDatetime(row["time"])
        snapshot.vitals.append(
            pb.VitalSign(
                metric_code=row["metric_code"],
                value=row["value"],
                unit=unit,
                measured_at=vt,
            )
        )

    # ------------------------------------------------------------------
    # Active drugs — infusions started before sim_time and not stopped
    # ------------------------------------------------------------------
    drug_rows = await store.fetch(
        """
        SELECT DISTINCT ON (drug_name)
               drug_name, dose, unit, route, ordered_by, time
        FROM mar
        WHERE patient_id = $1 AND time <= $2 AND route IN ('IV', 'infusion')
        ORDER BY drug_name, time DESC
        """,
        patient_id,
        sim_time,
    )
    for row in drug_rows:
        dt = Timestamp()
        dt.FromDatetime(row["time"])
        snapshot.active_drugs.append(
            pb.ActiveDrug(
                drug_name=row["drug_name"],
                current_rate=row["dose"],
                unit=row["unit"],
                started_at=dt,
            )
        )

    # ------------------------------------------------------------------
    # Pending messages — messages not yet acknowledged
    # Week 2: empty; Week 4 will wire LISTEN/NOTIFY or polling
    # ------------------------------------------------------------------

    return snapshot


def _hash_snapshot(sim_time: datetime, patient_id: int) -> str:
    """Deterministic version hash for causal tracking."""
    payload = f"{patient_id}:{sim_time.isoformat()}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _metric_unit(metric_code: str) -> str:
    units = {
        "hr": "bpm",
        "map": "mmHg",
        "sbp": "mmHg",
        "dbp": "mmHg",
        "rr": "bpm",
        "spo2": "%",
        "etco2": "mmHg",
        "temp_c": "°C",
        "cvp": "mmHg",
        "co": "L/min",
        "blood_volume_ml": "mL",
        "levo_rate": "mcg/min",
        "prop_rate": "mg/hr",
        "fent_rate": "mcg/hr",
    }
    return units.get(metric_code, "")
