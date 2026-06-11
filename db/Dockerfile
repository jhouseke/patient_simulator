# Multi-extension PostgreSQL 16 for ICU Patient Simulator
# Extensions: TimescaleDB + pgvector + Apache AGE
#
# Risk 1 from architecture spec: triple-extension stack validation.
# If this container fails under concurrent load, fallback to separate
# AGE container or recursive-CTE graph pattern.

FROM postgres:16-bookworm

LABEL org.opencontainers.image.title="patient-simulator-db"
LABEL org.opencontainers.image.description="PostgreSQL 16 + TimescaleDB + pgvector + Apache AGE"

# Build and runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    gnupg \
    lsb-release \
    wget \
    curl \
    git \
    build-essential \
    postgresql-server-dev-16 \
    libreadline-dev \
    zlib1g-dev \
    bison \
    flex \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------
# 1. TimescaleDB (apt repository)
# ------------------------------------------------------------------
RUN echo "deb https://packagecloud.io/timescale/timescaledb/debian/ $(lsb_release -c -s) main" \
    > /etc/apt/sources.list.d/timescaledb.list \
    && wget --quiet -O - https://packagecloud.io/timescale/timescaledb/gpgkey \
    | gpg --dearmor -o /etc/apt/trusted.gpg.d/timescaledb.gpg \
    && apt-get update \
    && apt-get install -y --no-install-recommends timescaledb2-postgresql-16 \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------
# 2. pgvector (PostgreSQL community apt repository)
# ------------------------------------------------------------------
RUN echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -c -s)-pgdg main" \
    > /etc/apt/sources.list.d/pgdg.list \
    && wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc \
    | apt-key add - \
    && apt-get update \
    && apt-get install -y --no-install-recommends postgresql-16-pgvector \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------
# 3. Apache AGE (build from release source)
# ------------------------------------------------------------------
ENV AGE_VERSION=1.5.0
RUN cd /tmp \
    && wget -q "https://github.com/apache/age/archive/refs/tags/PG16/v${AGE_VERSION}.tar.gz" \
    -O age.tar.gz \
    && tar -xzf age.tar.gz \
    && cd "age-PG16-v${AGE_VERSION}" \
    && make \
    && make install \
    && cd / \
    && rm -rf /tmp/age*

# ------------------------------------------------------------------
# 4. Runtime tuning for three extensions
# ------------------------------------------------------------------
# TimescaleDB and AGE both require shared_preload_libraries.
# pgvector does not. Ordering: timescaledb first (planner hooks),
# then age (graph engine).
RUN echo "shared_preload_libraries = 'timescaledb,age'" \
    >> /usr/share/postgresql/postgresql.conf.sample

# Increase shared_buffers for concurrent extension workloads.
# 256MB is a modest dev default; override via docker-compose for prod.
RUN echo "shared_buffers = 256MB" \
    >> /usr/share/postgresql/postgresql.conf.sample

# AGE requires standard_conforming_strings = on (default in PG16, but
# be explicit) and adds its own search_path entry via init scripts.

# Healthcheck script
COPY db/healthcheck.sh /usr/local/bin/db_healthcheck.sh
RUN chmod +x /usr/local/bin/db_healthcheck.sh
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=5 \
    CMD /usr/local/bin/db_healthcheck.sh

EXPOSE 5432
