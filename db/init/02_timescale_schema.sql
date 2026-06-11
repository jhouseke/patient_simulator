-- ------------------------------------------------------------------
-- 02_timescale_schema.sql
-- TimescaleDB hypertable schema for canonical numerical state.
-- ------------------------------------------------------------------

-- Narrow table for flexible vital sign types.
-- source: 'engine' | 'agent' | 'ipc'
CREATE TABLE IF NOT EXISTS vital_signs (
    time        TIMESTAMPTZ NOT NULL,
    patient_id  INT NOT NULL,
    metric_code TEXT NOT NULL,
    value       DOUBLE PRECISION NOT NULL,
    source      TEXT DEFAULT 'engine',
    UNIQUE (time, patient_id, metric_code, source)
);

SELECT create_hypertable('vital_signs', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_vitals_patient_metric
    ON vital_signs (patient_id, metric_code, time DESC);

-- ------------------------------------------------------------------
-- MAR (Medication Administration Record)
-- ------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS mar (
    time        TIMESTAMPTZ NOT NULL,
    patient_id  INT NOT NULL,
    drug_name   TEXT NOT NULL,
    dose        DOUBLE PRECISION NOT NULL,
    unit        TEXT NOT NULL,
    route       TEXT NOT NULL,
    ordered_by  TEXT NOT NULL,
    validated   BOOLEAN DEFAULT true
);

SELECT create_hypertable('mar', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_mar_patient_drug
    ON mar (patient_id, drug_name, time DESC);

-- ------------------------------------------------------------------
-- Agent action log (for observability and rollback auditing)
-- ------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS agent_actions (
    time            TIMESTAMPTZ NOT NULL,
    agent_id        TEXT NOT NULL,
    action_type     TEXT NOT NULL,
    payload         JSONB NOT NULL,
    emotion_vector  JSONB,           -- {anxiety, fatigue, confidence}
    based_on_hash   TEXT,
    was_rolled_back BOOLEAN DEFAULT false
);

SELECT create_hypertable('agent_actions', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_agent_actions_agent
    ON agent_actions (agent_id, time DESC);

CREATE INDEX IF NOT EXISTS idx_agent_actions_hash
    ON agent_actions (based_on_hash)
    WHERE based_on_hash IS NOT NULL;
