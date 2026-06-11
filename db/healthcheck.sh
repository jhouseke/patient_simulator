#!/usr/bin/env bash
set -eo pipefail

# Healthcheck for triple-extension PostgreSQL.
# Verifies core PostgreSQL is up AND all three extensions are loadable.

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" -c "
SELECT
    (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') AS timescaledb_ok,
    (SELECT 1 FROM pg_extension WHERE extname = 'vector')     AS pgvector_ok,
    (SELECT 1 FROM pg_extension WHERE extname = 'age')        AS age_ok;
" | grep -q "1 | 1 | 1"
