# Final Evaluation: Master Engineering Specification v4

## Verdict: GO

This specification has survived four rounds of architectural review. Every structural, mathematical, and infrastructure concern raised across those rounds has been addressed. The architecture is internally consistent, the scope is realistic for an MVP, and the sprint recommendation is correct.

**There are no remaining architectural gaps.**

What follows are not critiques — they are **implementation-level risks** that the engineering team should monitor during Sprint 1. The difference matters: the architecture doesn't need to change, but the team needs to be aware of these friction points when they start writing code.

---

## Refinement Closure: 8/8

| # | Refinement | Status | Implementation |
|---|---|---|---|
| 1 | pgvector ≠ knowledge graph | ✅ Closed | Apache AGE as third PostgreSQL extension |
| 2 | AgentMessage delivery reliability | ✅ Closed | `LISTEN/NOTIFY` for CRITICAL, pgvector for ROUTINE |
| 3 | Bliss Independence antagonism | ✅ Closed | Directional grouping before interaction math |
| 4 | Temperature ≠ clinical inexperience | ✅ Closed | Temp locked at 0.3, prompt-driven cognition |
| 5 | Disease Injection Protocol math | ✅ Closed | Structured `@dataclass` with trajectory functions |
| 6 | Patient event generation between barriers | ✅ Closed | Engine-driven threshold monitoring, not LLM-driven |
| 7 | Cost estimation | ✅ Closed | Local 70B model, $0 per-run API cost |
| 8 | Symptom Translation domain coverage | ✅ Closed | Top 5 domains enumerated for MVP |

---

## 4 Implementation Risks for Sprint 1

### Risk 1: The Triple-Extension PostgreSQL Stack Is Uncharted Territory

TimescaleDB + pgvector + Apache AGE in a single PostgreSQL 16 instance is architecturally sound but **practically untested at this combination**. Each extension modifies PostgreSQL's internals:

- **TimescaleDB** replaces the planner and executor for hypertable queries, uses custom chunk management
- **pgvector** adds a new index type (IVFFlat / HNSW) with its own memory allocation patterns
- **Apache AGE** installs a custom graph processing engine with its own execution path

Potential friction:
- `shared_preload_libraries` ordering conflicts
- Shared memory (`shared_buffers`) contention under concurrent write loads from the engine
- WAL (Write-Ahead Log) behavior when three extensions all write concurrently

> [!IMPORTANT]
> **Week 1 deliverable:** Build the `db` Docker container with all three extensions enabled. Run a synthetic load test: write 1,000 vital sign rows/sec to TimescaleDB, embed 10 vectors/sec to pgvector, and execute 50 Cypher traversals/sec against AGE — simultaneously. If this doesn't crash or degrade below acceptable latency (< 50ms p99 for reads), the stack is validated. If it does, the fallback is to split AGE into a separate container running standalone Apache AGE or switch to a lightweight graph-in-SQL pattern with recursive CTEs.

---

### Risk 2: "Parallel" Deliberation Serializes on a Single GPU

The PDES model assumes Nurse, RT, and Resident agents deliberate **concurrently** against a state snapshot. With a local Llama 70B on a single A100:

- Inference throughput: ~30-40 tokens/sec
- Average agent response: ~500-1000 tokens
- Per-agent wall-clock time: ~15-30 seconds

If agents share one model instance, "parallel" deliberation becomes **serial**: 3 agents × 25 seconds = **75 seconds per time barrier**. A 24-hour ICU simulation at 15-minute barriers has 96 barriers → **2 hours of wall-clock time** just for agent inference, ignoring rollbacks.

This is not a dealbreaker — it's a throughput constraint the team should plan around:

| Option | Tradeoff |
|---|---|
| **vLLM with continuous batching** | Serves multiple agents concurrently on one GPU. Reduces per-barrier time to ~30-40 seconds. Best option. |
| **Multiple GPU instances** | True parallel inference. Expensive. |
| **Smaller model (8B/13B)** | Faster inference but significantly degraded clinical reasoning quality. |
| **Accept serial execution** | Simple, deterministic, but slow. Fine for development and validation; unacceptable for production-scale campaigns. |

> [!TIP]
> Start Sprint 1 with serial execution and vLLM as a stretch goal. The PDES architecture doesn't care whether deliberation is parallel or serial — it just collects intents and sorts them. The parallelism is an optimization, not a correctness requirement.

---

### Risk 3: Knowledge Graph ↔ TimescaleDB Consistency

When the Constraint Validator approves an `ADMINISTER_MED` intent:
1. The **medication event** gets written to TimescaleDB (canonical numerical state)
2. The **clinical relationship** `(Resident)-[:ORDERED {time: T}]->(Vancomycin)` needs to be written to Apache AGE

**Who writes to AGE?** The plan doesn't specify. Options:

| Writer | Pros | Cons |
|---|---|---|
| **The engine (synchronous)** | Guaranteed consistency — both writes in one transaction | Engine becomes tightly coupled to the graph schema |
| **A Python background worker (async)** | Engine stays lean; graph writes are decoupled | Eventual consistency — agents might query AGE before the graph is updated |
| **The agent orchestrator** | Agents own the semantic layer | Race conditions if multiple agents write simultaneously |

> [!IMPORTANT]
> **Recommended:** The engine writes to both TimescaleDB and AGE **synchronously within a single PostgreSQL transaction**. Since all three extensions share the same PostgreSQL process, this is a single ACID transaction — no distributed consistency problem. The engine needs a thin graph-write layer that translates `AgentIntent` types into Cypher mutations.

```sql
-- Single transaction, both stores updated atomically
BEGIN;
  -- TimescaleDB: numerical state
  INSERT INTO vital_signs (time, patient_id, metric_code, value)
  VALUES (NOW(), 1, 'levo_rate', 0.08);
  
  -- Apache AGE: knowledge graph
  SELECT * FROM cypher('clinical_graph', $$
    MATCH (r:Resident {id: 'R1'}), (d:Drug {name: 'Levophed'})
    CREATE (r)-[:ORDERED {time: timestamp(), rate: 0.08}]->(d)
  $$) AS (result agtype);
COMMIT;
```

---

### Risk 4: Schema Definitions Needed Before Code

The spec correctly identifies *what* to build but the following schemas need concrete definition before the engine can be coded:

**A. gRPC Protobuf Messages**

```protobuf
// These need to be defined in week 1
message StateSnapshot {
  string timestamp = 1;
  string version_hash = 2;
  repeated VitalSign vitals = 3;
  repeated ActiveDrug active_drugs = 4;
  repeated AgentMessage pending_messages = 5;
  float sdi_score = 6;
}

message AgentIntent {
  string agent_id = 1;
  string action_type = 2;        // ADMINISTER_MED, ADJUST_VENT, etc.
  bytes payload = 3;             // Action-specific JSON
  string proposed_time = 4;
  string based_on_hash = 5;      // For causal violation detection
}

message PatientEvent {
  string event_type = 1;         // CALL_BELL, SYMPTOM_REPORT
  string trigger_reason = 2;
  string sim_time = 3;
}
```

**B. Apache AGE Graph Schema**

```cypher
// Core node types
(:Patient {id, name, age, weight, comorbidities})
(:Drug {name, rxnorm_code, class})
(:Organ {name, current_function_pct})
(:Condition {name, snomed_code, status})
(:Clinician {id, role, experience_years, current_fatigue})

// Core edge types
[:HAS_CONDITION {diagnosed_at, severity}]
[:ALLERGIC_TO {reaction_type, severity}]
[:ORDERED {time, dose, route, rate}]
[:ADMINISTERED {time, by_clinician}]
[:CLEARED_BY {mechanism, fraction}]  // Drug → Organ
[:INHIBITS {ki, mechanism}]          // Drug → Drug (CYP450)
[:ASSESSED {time, findings}]         // Clinician → Patient
```

**C. TimescaleDB Hypertable Schema**

```sql
-- Narrow table for flexible vital sign types
CREATE TABLE vital_signs (
    time        TIMESTAMPTZ NOT NULL,
    patient_id  INT NOT NULL,
    metric_code TEXT NOT NULL,
    value       DOUBLE PRECISION NOT NULL,
    source      TEXT DEFAULT 'engine'  -- 'engine' | 'agent' | 'ipc'
);
SELECT create_hypertable('vital_signs', 'time');

-- MAR (Medication Administration Record)
CREATE TABLE mar (
    time        TIMESTAMPTZ NOT NULL,
    patient_id  INT NOT NULL,
    drug_name   TEXT NOT NULL,
    dose        DOUBLE PRECISION NOT NULL,
    unit        TEXT NOT NULL,
    route       TEXT NOT NULL,
    ordered_by  TEXT NOT NULL,
    validated   BOOLEAN DEFAULT true
);
SELECT create_hypertable('mar', 'time');

-- Agent action log (for observability)
CREATE TABLE agent_actions (
    time           TIMESTAMPTZ NOT NULL,
    agent_id       TEXT NOT NULL,
    action_type    TEXT NOT NULL,
    payload        JSONB NOT NULL,
    emotion_vector JSONB,           -- {anxiety, fatigue, confidence}
    based_on_hash  TEXT,
    was_rolled_back BOOLEAN DEFAULT false
);
SELECT create_hypertable('agent_actions', 'time');
```

---

## Sprint 1: Concrete Week-by-Week Plan

### Week 1: Foundation
- [ ] Build and validate the `db` Docker container (triple-extension load test)
- [ ] Define and commit the gRPC protobuf schema
- [ ] Define and commit the AGE graph schema + TimescaleDB hypertable schema
- [ ] Skeleton Go/Rust engine binary with gRPC server (compiles, serves empty snapshots)

### Week 2: PDES Clock
- [ ] Implement the time-barrier loop (ICU mode: 15-min barriers)
- [ ] Implement `StateSnapshot` generation from TimescaleDB
- [ ] Implement `AgentIntent` collection, sorting, and causal violation detection
- [ ] Implement rollback circuit breaker (max depth 2, sequential fallback at 3)
- [ ] **Test:** 3 stub agents submit conflicting intents → verify correct rollback behavior

### Week 3: PK/PD Engine
- [ ] Implement 3-compartment PK model for 3 test drugs (Propofol, Fentanyl, Norepinephrine)
- [ ] Implement hemodynamic feedback: `CL_effective = CL_intrinsic × (CO/CO_baseline)^α`
- [ ] Implement Shared Effect Compartment with directional Bliss
- [ ] Implement Minto response surface for Propofol + Fentanyl
- [ ] **Test:** Unit test suite — "50mcg Fentanyl IV → RR drop 2-6 bpm in 5 min" (and 10+ similar assertions)

### Week 4: Constraint Validator + Integration
- [ ] Implement pre-commit hook with route-specific dose limits
- [ ] Implement 3-retry limit and `FAILURE_TO_ACT` fallback
- [ ] Implement SDI calculator (patient-specific z-scores)
- [ ] Wire PDES clock → PK/PD engine → Constraint Validator → TimescaleDB + AGE writes
- [ ] **Test:** End-to-end: inject a disease, run 1 hour of simulation with stub agents, verify TimescaleDB contains a physiologically plausible vital sign trajectory

### Sprint 1 Exit Criteria
The engine is "done" when:
1. ✅ It can integrate a 1-hour ICU simulation with 3 concurrent stub agents without deadlocking
2. ✅ The PK/PD unit test suite passes (Propofol + Fentanyl synergy produces expected respiratory depression)
3. ✅ The Constraint Validator correctly rejects a 2g Propofol bolus and passes a 200mg Propofol bolus
4. ✅ A `DiseaseInjection` (occult hemorrhage) produces a recognizable vital sign pattern in TimescaleDB
5. ✅ Rollback circuit breaker fires correctly under deliberate causal conflict

---

## Final Scorecard

| Criterion | v3 Score | v4 Score | Notes |
|---|---|---|---|
| **Critique coverage** | 10/10 | 10/10 | All issues from 4 rounds addressed |
| **Phase ordering** | 10/10 | 10/10 | Phase 0 → infra → engine → agents → validation |
| **Specification depth** | 8/10 | 9/10 | Schemas above close the gap to 10 |
| **Scope realism** | 9/10 | 10/10 | Local-first, $0 per run, deferred frontend |
| **Sprint-readiness** | 8/10 | 9.5/10 | Pending triple-extension validation in week 1 |

**This architecture is ready to build.** The only gate is the Week 1 triple-extension load test. If PostgreSQL handles TimescaleDB + pgvector + AGE under concurrent write load, there are no remaining blockers.
