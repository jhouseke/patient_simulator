-- ------------------------------------------------------------------
-- 03_age_graph_schema.sql
-- Apache AGE clinical knowledge graph schema.
-- Run inside the clinical_graph created in 01_extensions.sql.
-- ------------------------------------------------------------------

LOAD 'age';
SET search_path = ag_catalog, "$user", public;

-- Helper to avoid duplicate node creation.
-- We use MERGE-like patterns where possible; AGE PG16 v1.5 supports
-- basic MERGE. For idempotent schema init we guard with MATCH.

-- ------------------------------------------------------------------
-- Core node types
-- ------------------------------------------------------------------

-- (:Patient {id, name, age, weight, comorbidities})
SELECT * FROM cypher('clinical_graph', $$
    CREATE CONSTRAINT ON (p:Patient) ASSERT p.id IS UNIQUE
$$) AS (result agtype);

-- (:Drug {name, rxnorm_code, class})
SELECT * FROM cypher('clinical_graph', $$
    CREATE CONSTRAINT ON (d:Drug) ASSERT d.name IS UNIQUE
$$) AS (result agtype);

-- (:Organ {name})
SELECT * FROM cypher('clinical_graph', $$
    CREATE CONSTRAINT ON (o:Organ) ASSERT o.name IS UNIQUE
$$) AS (result agtype);

-- (:Condition {name, snomed_code})
SELECT * FROM cypher('clinical_graph', $$
    CREATE CONSTRAINT ON (c:Condition) ASSERT c.snomed_code IS UNIQUE
$$) AS (result agtype);

-- (:Clinician {id, role})
SELECT * FROM cypher('clinical_graph', $$
    CREATE CONSTRAINT ON (cl:Clinician) ASSERT cl.id IS UNIQUE
$$) AS (result agtype);

-- ------------------------------------------------------------------
-- Seed minimal MVP graph (one test patient, core drugs, organs)
-- ------------------------------------------------------------------

SELECT * FROM cypher('clinical_graph', $$
    MERGE (p:Patient {id: 'P1', name: 'Test Patient', age: 65, weight: 70, comorbidities: 'HTN,DM'})
    MERGE (d1:Drug {name: 'Propofol', rxnorm_code: '8302', class: 'sedative'})
    MERGE (d2:Drug {name: 'Fentanyl', rxnorm_code: '4337', class: 'opioid'})
    MERGE (d3:Drug {name: 'Norepinephrine', rxnorm_code: '7248', class: 'vasopressor'})
    MERGE (d4:Drug {name: 'Vancomycin', rxnorm_code: '11124', class: 'antibiotic'})
    MERGE (o1:Organ {name: 'Heart', current_function_pct: 100})
    MERGE (o2:Organ {name: 'Lungs', current_function_pct: 100})
    MERGE (o3:Organ {name: 'Kidneys', current_function_pct: 100})
    MERGE (o4:Organ {name: 'Liver', current_function_pct: 100})
    MERGE (o5:Organ {name: 'Brain', current_function_pct: 100})
    MERGE (c1:Condition {name: 'Hypertension', snomed_code: '38341003', status: 'active'})
    MERGE (c2:Condition {name: 'Diabetes Mellitus Type 2', snomed_code: '44054006', status: 'active'})
    MERGE (r1:Clinician {id: 'R1', role: 'Resident', experience_years: 2, current_fatigue: 0.3})
    MERGE (n1:Clinician {id: 'N1', role: 'Nurse', experience_years: 8, current_fatigue: 0.2})
    MERGE (rt1:Clinician {id: 'RT1', role: 'RespiratoryTherapist', experience_years: 5, current_fatigue: 0.15})
$$) AS (result agtype);

-- ------------------------------------------------------------------
-- Seed edges for MVP test patient
-- ------------------------------------------------------------------

SELECT * FROM cypher('clinical_graph', $$
    MATCH (p:Patient {id: 'P1'}),
          (c1:Condition {snomed_code: '38341003'}),
          (c2:Condition {snomed_code: '44054006'})
    MERGE (p)-[:HAS_CONDITION {diagnosed_at: timestamp(), severity: 'moderate'}]->(c1)
    MERGE (p)-[:HAS_CONDITION {diagnosed_at: timestamp(), severity: 'controlled'}]->(c2)
$$) AS (result agtype);

-- PK relationships: clearance organs for test drugs.
SELECT * FROM cypher('clinical_graph', $$
    MATCH (d1:Drug {name: 'Propofol'}),
          (d2:Drug {name: 'Fentanyl'}),
          (d3:Drug {name: 'Norepinephrine'}),
          (d4:Drug {name: 'Vancomycin'}),
          (liver:Organ {name: 'Liver'}),
          (kidneys:Organ {name: 'Kidneys'}),
          (lungs:Organ {name: 'Lungs'})
    MERGE (d1)-[:CLEARED_BY {mechanism: 'hepatic_glucuronidation', fraction: 0.98}]->(liver)
    MERGE (d2)-[:CLEARED_BY {mechanism: 'hepatic_CYP3A4', fraction: 0.75}]->(liver)
    MERGE (d2)-[:CLEARED_BY {mechanism: 'renal_unchanged', fraction: 0.10}]->(kidneys)
    MERGE (d3)-[:CLEARED_BY {mechanism: 'hepatic_MONOamine_oxidase', fraction: 0.90}]->(liver)
    MERGE (d4)-[:CLEARED_BY {mechanism: 'renal_unchanged', fraction: 0.90}]->(kidneys)
$$) AS (result agtype);

-- Drug-drug interaction: CYP3A4 inhibition (simplified for MVP).
SELECT * FROM cypher('clinical_graph', $$
    MATCH (propofol:Drug {name: 'Propofol'}),
          (fentanyl:Drug {name: 'Fentanyl'})
    MERGE (propofol)-[:INHIBITS {ki: 50.0, mechanism: 'CYP3A4_competitive'}]->(fentanyl)
$$) AS (result agtype);
