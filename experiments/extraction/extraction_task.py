from experiments.corenlp.openie_wrapper import StanfordOpenIE
from experiments.db.db_gate import DBGate
from experiments.extraction.extraction_filters import Extractor
from spacy.en import English


# todo: TaskManager
# governs: nlp, dbgate, oie server, oie extractor
# provides: 'tasks', i.e.: oie extraction, final extraction


def extract_task():
    nlp = English()
    db = DBGate(nlp)
    extractor = Extractor(nlp)
    extractor_ver = extractor.version
    oie = StanfordOpenIE()  # Stanford CoreNLP server must be running
    oie_ver = oie.version

    # Extraction subtask
    #
    # for sent_id, sent in db.get_spans():
    #     for extraction in oie.process(sent):
    #         extraction_id = db.add_extraction(extraction.text, sent_id, oie_ver)

    # Joint subtask (text -> sents -> raw_extractions)
    #
    for text_id, text in db.get_raw_texts():
        for pos, sent in enumerate(text.sents()):
            sent_id = db.add_span(sent.text, pos, text_id)
            for extraction in oie.process(sent):
                extraction_id = db.add_extraction(extraction.text, sent_id, oie_ver)

    # Thing extraction subtask (extractions -> {entities, relations, attributes})
    #
    for extraction_id, raw_extraction in db.get_extractions():
        for entities, relations, attrs in extractor.process(raw_extraction):
            for entity_span, entity in entities:
                db.add_entity(entity_span, entity.text, extraction_id, extractor_ver)
            for relation_span, relation in relations:
                db.add_relation(relation_span, relation.text, extraction_id, extractor_ver)
            for attr_name, (attr_span, attr) in attrs.items():
                db.add_attribute(attr_span, attr.text, extraction_id, attr_name, extractor_ver)

