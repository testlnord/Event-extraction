import logging as log
import json
from spacy.en import English
from experiments.event_with_offsets import EventOffsets


class PreprocessJsonOpenIEExtractions:
    """Parses json representatoins of OpenIE extractions and makes events from them."""

    def __init__(self, nlp, nb_most_confident=1):
        """nb_most_confident: yield at max that number of events per extraction
        events with high confidence preferred. if -1 then return all possible"""
        self.nb_most_confident = nb_most_confident
        self.nlp = nlp

    def events(self, json_event):
        ces = sorted(self._parse(json_event), key=lambda x: x[0])
        last_index = len(ces) if self.nb_most_confident == -1 else min(len(ces), self.nb_most_confident)
        ces = ces[:last_index]
        for conf, ev in ces:
            yield ev

    # todo: parse TIME, LOCATION and use context and confidence
    # todo: some problem with action_offsets. only the first offset in action_offsets is used. somewhere is bug...
    def _parse(self, text):
        """Yields parsed events"""
        j = json.loads(text)

        sentence = j['sentence']
        for i in j['instances']:
            # not using this for now
            confidence = i['confidence']
            e = i['extraction']

            entity1 = e['arg1']['val']
            action = e['rel']['val']
            entity2 = [ee['val'] for ee in e['arg2s']]
            entity2 = ' '.join(entity2)

            # parse Time and Location entities
            # ignoring them for now
            entity2 = str.replace(entity2, 'T:', '')
            entity2 = str.replace(entity2, 'L:', '')

            # not using this for now
            # context = e['context']
            # if context:
            #     pass

            entity1_offs = e['arg1']['offsets']
            action_offs = e['rel']['offsets']
            entity2_offs = [offset for ee in e['arg2s'] for offset in ee['offsets']]
            offset_event = EventOffsets(entity1_offs, entity2_offs, action_offs, sentence,
                                        entity1, entity2, action)

            yield confidence, offset_event


def test():
    path = "../samples-json-relations.txt"
    nlp = English()
    preprocessor = PreprocessJsonOpenIEExtractions(nlp, 2)
    with open(path) as f:
        for i, json_e in enumerate(f):
            for e in preprocessor.events(json_e):
                print(i, repr(e))


if __name__ == "__main__":
    log.basicConfig(format='%(levelname)s:%(message)s', level=log.DEBUG)
    test()

