CREATE OR REPLACE VIEW entities_with_f AS
SELECT f_entity_uid, entity_uid, f_entity_text, entity_text
FROM entities NATURAL LEFT JOIN entitiy_sets NATURAL LEFT JOIN f_entities
ORDER BY f_entity_uid, entity_text;


CREATE OR REPLACE VIEW relations_with_f AS
SELECT f_relation_uid, relatioin_uid, f_relation_text, relation_text
FROM relations NATURAL LEFT JOIN relation_sets NATURAL LEFT JOIN f_relations
ORDER BY f_relation_uid, relation_text;


CREATE OR REPLACE VIEW texts_spans AS
SELECT text_uid, dt_extracted, text_pos, span_uid, span_text
FROM spans NATURAL LEFT JOIN raw_texts
ORDER BY text_uid, text_pos;


CREATE OR REPLACE VIEW spans_extractions AS
SELECT *
FROM extractions NATURAL LEFT JOIN spans
ORDER BY span_uid, extraction_uid;


CREATE OR REPLACE VIEW extractions_full AS
SELECT *
FROM extractions
NATURAL LEFT JOIN entities
NATURAL LEFT JOIN relations
NATURAL LEFT JOIN attr_date
NATURAL LEFT JOIN attr_version
NATURAL LEFT JOIN attr_location
ORDER BY span_uid, extraction_uid;

