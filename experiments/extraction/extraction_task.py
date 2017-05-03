from experiments.corenlp.openie_wrapper import StanfordOpenIE
from experiments.db.db_gate import DBGate
from experiments.extraction.extraction_filters import Extractor, ExtractionFilter
from spacy.en import English


class TaskManager:
    def __init__(self, nlp, db: DBGate, oie: StanfordOpenIE):
        self.nlp = nlp
        self.db = db
        self.oie = oie
        self.extractor = Extractor(nlp)
        self.efilter = ExtractionFilter()

    def extract_extractions(self):
        """
        Extract sentences from texts, store them in db and extract extractions from sentences and store them.
        (text -> sentences -> extractions)
        """
        for text_id, text in self.db.get_raw_texts():
            for pos, sent in enumerate(text.sents):
                sent_id = self.db.add_span(sent, pos, text_id)
                for extraction in self.oie.process(sent):
                    extraction_id = self.db.add_extraction(extraction, sent_id, self.oie.version)
                    yield extraction_id

    def process_extractions(self):
        """
        Extract entities, relations and attributes from extractions and store them in db.
        (extractions -> {entities, relations, attributes})
        """
        for extraction_id, extraction in self.db.get_extractions():
            for entities, relations, attrs in self.extractor.process(extraction):
                if self.efilter.process(entities, relations, attrs):
                    for part, part_ents in entities.items():
                        for entity_span, entity in part_ents:
                            self.db.add_entity(entity_span, entity.text, extraction_id, self.extractor.version)
                    for relation_span, relation in relations:
                        self.db.add_relation(relation_span, relation.text, extraction_id, self.extractor.version)
                    for attr_name, (attr_span, attr) in attrs.items():
                        self.db.add_attribute(attr_span, attr.text, extraction_id, attr_name, self.extractor.version)


def test_add_samples_to_db(db):
    with open('../data/samples.txt') as f:
        text = f.read()
        text_uid = db.add_raw_text(text, source_id=1)
        print('text added to db; text_uid: {}'.format(text_uid))


if __name__ == "__main__":
    nlp = English()
    db = DBGate(nlp)
    oie = StanfordOpenIE()  # Stanford CoreNLP server must be running
    taskman = TaskManager(nlp, db, oie)

    for i, (id, extraction) in enumerate(db.get_extractions()):
        print(i, id, extraction)
