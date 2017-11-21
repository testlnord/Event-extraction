CREATE TABLE IF NOT EXISTS attr_version (
    extraction_uid int NOT NULL REFERENCES extractions,

    attr_span int4range NOT NULL CHECK (NOT isempty(attr_span)),
    attr_text text NOT NULL CHECK (attr_text <> ''),

    extractor_ver int NOT NULL CHECK (extractor_ver >= 0),
    dt_extracted timestamp NOT NULL DEFAULT now(),
    attr_uid serial PRIMARY KEY
);


CREATE TABLE IF NOT EXISTS attr_date (
    extraction_uid int NOT NULL REFERENCES extractions,

    attr_span int4range NOT NULL CHECK (NOT isempty(attr_span)),
    attr_text text NOT NULL CHECK (attr_text <> ''),

    extractor_ver int NOT NULL CHECK (extractor_ver >= 0),
    dt_extracted timestamp NOT NULL DEFAULT now(),
    attr_uid serial PRIMARY KEY
);


CREATE TABLE IF NOT EXISTS attr_location (
    extraction_uid int NOT NULL REFERENCES extractions,

    attr_span int4range NOT NULL CHECK (NOT isempty(attr_span)),
    attr_text text NOT NULL CHECK (attr_text <> ''),

    extractor_ver int NOT NULL CHECK (extractor_ver >= 0),
    dt_extracted timestamp NOT NULL DEFAULT now(),
    attr_uid serial PRIMARY KEY
);

