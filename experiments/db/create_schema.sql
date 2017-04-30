
-- some representation of source, i.e. description
CREATE TABLE IF NOT EXISTS sources (
    source_type text NOT NULL CHECK (source_type <> ''), -- e.g. 'article' or 'manual'
    source text NOT NULL CHECK (source <> ''),

    source_uid int PRIMARY KEY
);


-- TODO: add extractor version?
CREATE TABLE IF NOT EXISTS raw_texts (
    source_uid int NOT NULL REFERENCES sources, -- if we'll need add texts manually, then just use special source type, e.g. 'manual'

    raw_text text NOT NULL CHECK (raw_text <> ''),
    dt_extracted timestamp NOT NULL DEFAULT now(),

    text_uid int PRIMARY KEY
);


CREATE TABLE IF NOT EXISTS spans (
    text_uid int REFERENCES raw_texts NOT NULL,
    text_pos int NOT NULL CHECK (text_pos >= 0),

    span_text text NOT NULL CHECK (span_text <> ''),

    span_uid int PRIMARY KEY
);


CREATE TABLE IF NOT EXISTS extractions (
    span_uid int NOT NULL REFERENCES spans,

    subject int4range NOT NULL CHECK (NOT isempty(subject)),
    relation int4range NOT NULL CHECK (NOT isempty(relation)),
    object_min int4range NOT NULL CHECK (NOT isempty(object_min)),
    object_max int4range NOT NULL CHECK (NOT isempty(object_max)),

    extractor_ver int NOT NULL CHECK (extractor_ver >= 0),
    dt_extracted timestamp NOT NULL DEFAULT now(),
    extraction_uid int PRIMARY KEY
);


---- Raw Part ----
-- textual representation is for convenience, users should not rely on that


CREATE TABLE IF NOT EXISTS entities (
    extraction_uid int NOT NULL REFERENCES extractions,

    entity_span int4range NOT NULL CHECK (NOT isempty(entity_span)),
    entity_text text NOT NULL CHECK (entity_text <> ''),

    extractor_ver int NOT NULL CHECK (extractor_ver >= 0),
    dt_extracted timestamp NOT NULL DEFAULT now(),
    entity_uid int PRIMARY KEY
);


CREATE TABLE IF NOT EXISTS relations (
    extraction_uid int NOT NULL REFERENCES extractions,

    relation_span int4range NOT NULL CHECK (NOT isempty(relation_span)),
    relation_text text NOT NULL CHECK (relation_text <> ''),

    extractor_ver int NOT NULL CHECK (extractor_ver >= 0),
    dt_extracted timestamp NOT NULL DEFAULT now(),
    relation_uid int PRIMARY KEY
);


---- Structured Part ----
-- TODO: add extractor version?


CREATE TABLE IF NOT EXISTS f_entities (
    f_entity_text text NOT NULL CHECK (f_entity_text <> ''),

    f_entity_uid int PRIMARY KEY
);


CREATE TABLE IF NOT EXISTS f_relations (
    f_relation_text text NOT NULL CHECK (f_relation_text <> ''),

    f_relation_uid int PRIMARY KEY
);


CREATE TABLE IF NOT EXISTS entity_sets (
    entity_uid int REFERENCES entities NOT NULL,
    f_entity_uid int REFERENCES f_entities NOT NULL
);


CREATE TABLE IF NOT EXISTS relation_sets (
    relation_uid int REFERENCES relations NOT NULL,
    f_relation_uid int REFERENCES f_relations NOT NULL
);

