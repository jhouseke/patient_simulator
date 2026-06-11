#!/usr/bin/env python3
"""
Week 1 Gate: Triple-Extension PostgreSQL Load Test

Validates that TimescaleDB + pgvector + Apache AGE can coexist under
concurrent write load. Run this AFTER `make db-up` has healthy status.

Target load (simultaneous):
  - 1,000 vital_sign rows/sec  → TimescaleDB
  - 10 embedding vectors/sec   → pgvector
  - 50 Cypher traversals/sec   → Apache AGE

Acceptance: p99 read latency < 50ms, no crashes, no WAL errors.
"""

from __future__ import annotations

import asyncio
import os
import random
import statistics
import time
from datetime import datetime, timezone

import asyncpg

DSN = os.getenv(
    "DATABASE_URL",
    "postgresql://psim:psim_dev@localhost:5432/patient_simulator",
)

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
DURATION_SEC = 30
VITALS_RATE = 1_000       # rows/sec
VECTOR_RATE = 10          # inserts/sec
CYPHER_RATE = 50          # traversals/sec
READ_LATENCY_SAMPLES = 100

# ------------------------------------------------------------------
# Workers
# ------------------------------------------------------------------


async def _ensure_table(conn: asyncpg.Connection) -> None:
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS load_test_embeddings (
            id SERIAL PRIMARY KEY,
            time TIMESTAMPTZ DEFAULT NOW(),
            vec vector(128)
        )
    """)


async def vitals_writer(pool: asyncpg.Pool, duration: float, counter: list[int]) -> None:
    """Firehose vital_signs inserts."""
    end = time.monotonic() + duration
    metric_codes = ["hr", "map", "rr", "spo2", "temp_c", "cvp", "co"]
    while time.monotonic() < end:
        rows = [
            (
                datetime.now(timezone.utc),
                1,
                random.choice(metric_codes),
                random.gauss(80, 10),
                "load_test",
            )
            for _ in range(50)
        ]
        async with pool.acquire() as conn:
            await conn.executemany(
                "INSERT INTO vital_signs (time, patient_id, metric_code, value, source) VALUES ($1, $2, $3, $4, $5)",
                rows,
            )
        counter[0] += len(rows)
        await asyncio.sleep(0)


async def vector_writer(pool: asyncpg.Pool, duration: float, counter: list[int]) -> None:
    """Insert random 128-dim vectors into pgvector table."""
    end = time.monotonic() + duration
    while time.monotonic() < end:
        vec = [random.random() for _ in range(128)]
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO load_test_embeddings (vec) VALUES ($1)",
                vec,
            )
        counter[0] += 1
        await asyncio.sleep(1.0 / VECTOR_RATE)


async def cypher_traverser(pool: asyncpg.Pool, duration: float, counter: list[int]) -> None:
    """Run simple Cypher traversals against AGE clinical_graph."""
    end = time.monotonic() + duration
    query = """
        SELECT * FROM cypher('clinical_graph', $$
            MATCH (p:Patient {id: 'P1'})-[:HAS_CONDITION]->(c:Condition)
            RETURN c.name
        $$) AS (condition agtype)
    """
    while time.monotonic() < end:
        async with pool.acquire() as conn:
            await conn.execute("LOAD 'age'")
            await conn.execute('SET search_path = ag_catalog, "$user", public')
            await conn.fetch(query)
        counter[0] += 1
        await asyncio.sleep(0)


async def latency_checker(pool: asyncpg.Pool, duration: float, samples: list[float]) -> None:
    """Sample read latency from vital_signs hypertable."""
    end = time.monotonic() + duration
    while time.monotonic() < end:
        t0 = time.monotonic()
        async with pool.acquire() as conn:
            await conn.fetch(
                "SELECT * FROM vital_signs WHERE patient_id = 1 ORDER BY time DESC LIMIT 10"
            )
        samples.append((time.monotonic() - t0) * 1000)
        await asyncio.sleep(duration / READ_LATENCY_SAMPLES)


# ------------------------------------------------------------------
# Orchestrator
# ------------------------------------------------------------------


async def main() -> None:
    print("=" * 60)
    print("Triple-Extension Load Test")
    print(f"Target: {VITALS_RATE} vitals/s | {VECTOR_RATE} vectors/s | {CYPHER_RATE} cypher/s")
    print(f"Duration: {DURATION_SEC}s")
    print("=" * 60)

    pool = await asyncpg.create_pool(DSN, min_size=5, max_size=20)
    async with pool.acquire() as conn:
        await _ensure_table(conn)

    vitals_count = [0]
    vector_count = [0]
    cypher_count = [0]
    latencies: list[float] = []

    t_start = time.monotonic()
    await asyncio.gather(
        vitals_writer(pool, DURATION_SEC, vitals_count),
        vector_writer(pool, DURATION_SEC, vector_count),
        cypher_traverser(pool, DURATION_SEC, cypher_count),
        latency_checker(pool, DURATION_SEC, latencies),
    )
    elapsed = time.monotonic() - t_start

    await pool.close()

    # Report
    print("\n--- Results ---")
    print(f"Elapsed wall time: {elapsed:.1f}s")
    print(f"Vital rows written:   {vitals_count[0]:,}  ({vitals_count[0]/elapsed:,.0f}/s)")
    print(f"Vector rows written:  {vector_count[0]:,}  ({vector_count[0]/elapsed:,.0f}/s)")
    print(f"Cypher traversals:    {cypher_count[0]:,}  ({cypher_count[0]/elapsed:,.0f}/s)")

    if latencies:
        latencies.sort()
        p50 = statistics.median(latencies)
        p99 = latencies[int(len(latencies) * 0.99)]
        print(f"Read latency  p50: {p50:.2f}ms")
        print(f"Read latency  p99: {p99:.2f}ms")
        if p99 < 50:
            print("\n✅ PASS — p99 < 50ms")
        else:
            print("\n❌ FAIL — p99 >= 50ms (consider fallback architecture)")
    else:
        print("No latency samples collected.")


if __name__ == "__main__":
    asyncio.run(main())
