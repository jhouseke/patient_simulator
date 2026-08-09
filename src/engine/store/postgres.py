"""
Thin PostgreSQL access layer.

Wraps asyncpg for TimescaleDB + pgvector queries.
Apache AGE Cypher queries are sent through the same pool since all
three extensions live in one PostgreSQL process (Week 1 gate).
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Sequence

import asyncpg


class PostgresStore:
    """Async connection pool with helpers for the triple-extension stack."""

    def __init__(self, dsn: str | None = None) -> None:
        self.dsn = dsn or self._default_dsn()
        self._pool: asyncpg.Pool | None = None

    @staticmethod
    def _default_dsn() -> str:
        host = os.getenv("PGHOST", "localhost")
        port = os.getenv("PGPORT", "5432")
        user = os.getenv("PGUSER", "psim")
        password = os.getenv("PGPASSWORD", "psim_dev")
        db = os.getenv("PGDATABASE", "patient_simulator")
        return f"postgresql://{user}:{password}@{host}:{port}/{db}"

    async def connect(self) -> None:
        self._pool = await asyncpg.create_pool(self.dsn, min_size=2, max_size=10)

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None

    async def fetch(self, query: str, *args: Any) -> list[asyncpg.Record]:
        if not self._pool:
            raise RuntimeError("PostgresStore not connected")
        async with self._pool.acquire() as conn:
            return await conn.fetch(query, *args)

    async def execute(self, query: str, *args: Any) -> str:
        if not self._pool:
            raise RuntimeError("PostgresStore not connected")
        async with self._pool.acquire() as conn:
            return await conn.execute(query, *args)

    # ------------------------------------------------------------------
    # Vital signs
    # ------------------------------------------------------------------

    async def insert_vitals(
        self,
        patient_id: int,
        time: datetime,
        vitals: dict[str, float],
        source: str = "engine",
    ) -> None:
        """Upsert one timestamp's worth of vitals."""
        await self.insert_vitals_batch(patient_id, [(time, vitals)], source=source)

    async def insert_vitals_batch(
        self,
        patient_id: int,
        samples: Sequence[tuple[datetime, dict[str, float]]],
        source: str = "engine",
    ) -> None:
        """Upsert many timestamps in one round-trip.

        samples: [(time, {metric_code: value}), ...]

        The batched form exists because charting runs at ~5 s cadence: a
        15-minute barrier produces 180 sample dicts, and per-sample round-trips
        would dominate barrier wall time.
        """
        records = [
            (t, patient_id, code, float(value), source)
            for t, vitals in samples
            for code, value in vitals.items()
        ]
        if not records:
            return
        if not self._pool:
            raise RuntimeError("PostgresStore not connected")
        async with self._pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO vital_signs (time, patient_id, metric_code, value, source)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (time, patient_id, metric_code, source)
                DO UPDATE SET value = EXCLUDED.value
                """,
                records,
            )

    async def cypher(self, graph_name: str, query: str) -> list[asyncpg.Record]:
        """
        Execute a Cypher query against Apache AGE.

        AGE requires the 'age' extension to be loaded in the session.
        We do this via SET commands before each Cypher call.
        """
        if not self._pool:
            raise RuntimeError("PostgresStore not connected")
        async with self._pool.acquire() as conn:
            await conn.execute("LOAD 'age'")
            await conn.execute("SET search_path = ag_catalog, \"$user\", public")
            wrapped = f"""
                SELECT * FROM cypher('{graph_name}', $$
                    {query}
                $$) AS (result agtype)
            """
            return await conn.fetch(wrapped)
