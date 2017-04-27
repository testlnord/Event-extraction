
CREATE TABLE sources (
    source text NOT NULL CHECK (source <> ''),

    source_uid int PRIMARY KEY
);


CREATE TABLE raw_texts (
    raw_text text NOT NULL CHECK (raw_text <> ''),
    dt_extracted timestamp NOT NULL DEFAULT now(),

    text_uid int PRIMARY KEY,
    source_uid int NOT NULL REFERENCES sources
);
    

CREATE TABLE spans (
    span_text text NOT NULL CHECK (span_text <> ''),

    span_uid int PRIMARY KEY,
    text_uid int REFERENCES raw_texts NOT NULL,
    text_pos int CHECK (text_pos >= 0) DEFAULT 0
    --CONSTRAINT valid_text CHECK (NOT ((text_uid IS NULL) AND (text_pos <> 0)))
);


CREATE TABLE extractions (
    subject int4range NOT NULL CHECK (NOT isempty(subject)),
    relation int4range NOT NULL CHECK (NOT isempty(relation)),
    object_min int4range NOT NULL CHECK (NOT isempty(object_min)),
    object_max int4range NOT NULL CHECK (NOT isempty(object_max)),
    dt_extracted timestamp NOT NULL DEFAULT now(),

    extraction_uid int PRIMARY KEY,
    span_uid int NOT NULL REFERENCES spans
);
    

CREATE OR REPLACE VIEW extractions_full AS
SELECT * FROM extractions NATURAL LEFT JOIN spans;


--TODO: add textual representation for convenience?
---- Raw Part ----

CREATE TABLE entities (
    entity_span int4range NOT NULL CHECK (NOT isempty(entity_span)),
    dt_extracted timestamp NOT NULL DEFAULT now(),

    entity_uid int PRIMARY KEY,
    extraction_uid int NOT NULL REFERENCES extractions
);

CREATE TABLE relations (
    relation_span int4range NOT NULL CHECK (NOT isempty(relation_span)),
    dt_extracted timestamp NOT NULL DEFAULT now(),

    relation_uid int PRIMARY KEY,
    extraction_uid int NOT NULL REFERENCES extractions
);


CREATE TABLE attr_version (
    span int4range NOT NULL CHECK (NOT isempty(relation_span)),
    dt_extracted timestamp NOT NULL DEFAULT now(),

    attr_uid int PRIMARY KEY,
    extraction_uid int NOT NULL REFERENCES extractions
);

CREATE TABLE attr_date (
    span int4range NOT NULL CHECK (NOT isempty(relation_span)),
    dt_extracted timestamp NOT NULL DEFAULT now(),

    attr_uid int PRIMARY KEY,
    extraction_uid int NOT NULL REFERENCES extractions
);

CREATE TABLE attr_location (
    span int4range NOT NULL CHECK (NOT isempty(relation_span)),
    dt_extracted timestamp NOT NULL DEFAULT now(),

    attr_uid int PRIMARY KEY,
    extraction_uid int NOT NULL REFERENCES extractions
);


---- Structured Part ----

CREATE TABLE f_entities (
    f_entity_text text NOT NULL CHECK (entity_text <> ''),

    f_entity_uid int PRIMARY KEY,
    entity_uid int REFERENCES entities
);


CREATE TABLE f_relations (
    f_relation_text text NOT NULL CHECK(relation_text <> ''),

    f_relation_uid int PRIMARY KEY,
    relation_uid int REFERENCES relations
);


