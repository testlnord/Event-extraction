import json
from spacy.en import English
from event import Event


def parse_extrs(file):
    with open(file, 'r') as f:
        for line in f.readlines():
            yield json.loads(line)


# todo: parsing TIME, LOCATION and use context and confidence
def to_events(parsed_json):
    j = parsed_json
    events = []
    sentence = j['sentence']
    confidence = 0
    for i in j['instances']:
        # not using this for now
        confidence = i['confidence']

        for e in i['extraction']:
            entity1 = e['arg1']
            entity2 = e['arg2s']
            action = e['rel']

            # parse Time and Location entities
            # ignoring them for now
            entity2 = str.replace(entity2, 'T:', '')
            entity2 = str.replace(entity2, 'L:', '')

            # not using this for now
            context = e['context']
            if context:
                pass

            event = Event(entity1=entity1, entity2=entity2, action=action, sentence=sentence)
            events.append(event)
    return events


def parse_and_output_json_events(path):
    sent = 0
    for parsed in parse_extrs(path):
        # print(parsed)
        sent += 1
        extr = 0
        for event in to_events(parsed):
            extr += 1
            print("{}.{}".format(sent, extr))
            print(event)


# todo: analyze dep trees patterns formed by these extractions
def count_patterns(events, nlp):
    for ev in events:
        sent = nlp(events.sentence)
    pass


if __name__ == "__main__":
    input_path = "samples-relations.txt"
    parse_and_output_json_events(input_path)
