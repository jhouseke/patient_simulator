-- ------------------------------------------------------------------
-- 01_extensions.sql
-- Load all three PostgreSQL extensions idempotently.
-- ------------------------------------------------------------------

-- TimescaleDB: time-series engine
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- pgvector: vector similarity search for agent memory / embeddings
CREATE EXTENSION IF NOT EXISTS vector;

-- Apache AGE: graph database for clinical knowledge graph
LOAD 'age';
SET search_path = ag_catalog, "$user", public;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM ag_graph WHERE name = 'clinical_graph') THEN
        PERFORM create_graph('clinical_graph');
    END IF;
END $$;
