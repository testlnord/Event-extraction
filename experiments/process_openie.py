import json
from event import Event


def parse_extractions(file):
    with open(file, 'r') as f:
        for line in f.readlines():
            yield json.loads(line)


class EventOffsets(Event):
    def __init__(self, entity1_offs, action_offs, entity2_offs, sentence):
        self.sentence = sentence
        self.entity1_offsets = entity1_offs
        self.action_offsets = action_offs
        self.entity2_offsets = entity2_offs

        entity1 = self._substring(entity1_offs)
        entity2 = self._substring(entity2_offs)
        action = self._substring(action_offs)
        super().__init__(entity1=entity1, entity2=entity2, action=action, sentence=sentence)

    def _substring(self, offsets):
        sent = self.sentence
        s = [sent[o[0]:o[1]] for o in offsets]
        return ' '.join(s)

    @property
    def offsets(self):
        return self.entity1_offsets + self.action_offsets + self.entity2_offsets


# todo: parse TIME, LOCATION and use context and confidence
# todo: some problem with action_offsets. only the first offset in action_offsets is used. somewhere is bug...
def to_events(parsed_json):
    j = parsed_json
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

        event = Event(entity1=entity1, entity2=entity2, action=action, sentence=sentence)

        entity1 = e['arg1']['offsets']
        action = e['rel']['offsets']
        entity2 = [offset for ee in e['arg2s'] for offset in ee['offsets']]
        offset_event = EventOffsets(entity1, action, entity2, sentence)

        # yield confidence, event
        yield confidence, offset_event


def event_parser_generator(path):
    for parsed in parse_extractions(path):
        for confidence, event in to_events(parsed):
            yield confidence, event


# same as event_parser_generator, for convenience
def parse_and_output_json_events(path):
    sent = 0
    for parsed in parse_extractions(path):
        # print(parsed)
        sent += 1
        extr = 0
        for confidence, event in to_events(parsed):
            extr += 1
            print("{}.{}; confidence={}".format(sent, extr, confidence))
            print(event.offsets)
            # print(str.replace(str(event), "'", "’"))


# todo: analyze dep trees patterns formed by these extractions
def count_patterns(events, nlp):
    for ev in events:
        sent = nlp(events.sentence)
    pass


if __name__ == "__main__":
    input_path = "samples-json-relations.txt"
    parse_and_output_json_events(input_path)
