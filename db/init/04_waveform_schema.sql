-- Waveform storage: one row per (patient, channel, window). `samples` is a
-- fixed-duration array of REAL to avoid row-per-sample explosion — a 125 Hz
-- arterial line over a 6-hour case is 2.7M samples per channel.
CREATE TABLE IF NOT EXISTS waveform_chunks (
    time            TIMESTAMPTZ NOT NULL,   -- window start
    patient_id      INT NOT NULL,
    channel         TEXT NOT NULL,          -- ART | PLETH | CO2 | AWP | FLOW | VOL | ECG_II
    sample_rate_hz  REAL NOT NULL,
    duration_s      REAL NOT NULL,
    samples         REAL[] NOT NULL,
    UNIQUE (time, patient_id, channel)
);

SELECT create_hypertable('waveform_chunks', 'time', if_not_exists => TRUE);

ALTER TABLE waveform_chunks SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'patient_id, channel'
);

SELECT add_compression_policy('waveform_chunks', INTERVAL '7 days', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_waveform_patient_channel
    ON waveform_chunks (patient_id, channel, time DESC);
