import logging as log
import os
import re
import pickle

from experiments.ontology.data_structs import RelRecord


def process_record(text):
    lines = list(filter(None, text.split('\n')))
    data, rel = lines[:2]  # omitting 'Comment:' field if it is present
    rels = rel.split('(')
    if len(rels) > 1:
        rel = rels[0]
        direction = rels[1][:2] == 'e1'
    else:  # there's no direction, so, relation is None (NB: ad-hoc for these 2 datasets)
        direction = rel = None  # designates absence of relation between entities

    unquoted = data.split('"')
    i = int(unquoted[0].strip())  # number of data record
    data = ''.join(unquoted[1:-1])  # strip quotes

    parts = re.split('</?e(\d)>', data)
    tparts = list(map(str.strip, parts[::2]))
    if len(tparts) != 5:
        log.warning('bad data #{} in {}: delimiters for entities not found.'.format(i, data))
        return

    subj = tparts[1]
    objc = tparts[3]
    first = int(parts[1]) - 1  # num of entity (0 - subject, 1 - object)
    # Swap subject and object according to direction of relation
    if direction is not None and bool(first) == direction:  # if direction exists and not synced
        subj, objc = objc, subj

    context = ' '.join(filter(None, tparts))
    s0 = context.find(subj); s1 = s0 + len(subj)
    o0 = context.find(objc); o1 = o0 + len(objc)

    assert context[s0:s1] == subj
    assert context[o0:o1] == objc
    return i, RelRecord(rel, s0, s1, o0, o1, context)


def preprocess(input_path, output_path):
    with open(input_path) as f:
        records = f.read().split('\n\n')
        log.info('{} records in "{}"'.format(len(records), input_path))
    with open(output_path, 'wb') as f:
        total = 0
        for text in records:
            if text.strip():
                total += 1
                i, record = process_record(text)
                print('<{}>: <{}>'.format(i, record.context))
                print('<{}> <{}>'.format(record.subject_text, record.object_text))
                print('<{}> <{}>'.format(record.relation, record.direction))
                print()
                pickle.dump(record, f)
        print('TOTAL: {}'.format(total))


if __name__ == "__main__":
    log.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=log.INFO)

    data_dir = '/home/user/datasets/'
    kbp_dir = os.path.join(data_dir, 'kbp37')
    semeval_dir = os.path.join(data_dir, 'semeval2010')

    filenames = ['train', 'test', 'dev']
    ext = '.txt'
    for filename in filenames:
        input_path = os.path.join(kbp_dir, filename + ext)
        output_path = os.path.join(kbp_dir, filename + '.pck')
        preprocess(input_path, output_path)

    filenames = ['train', 'test']
    ext = '.txt'
    for filename in filenames:
        input_path = os.path.join(semeval_dir, filename + '.txt')
        output_path = os.path.join(semeval_dir, filename + '.pck')
        preprocess(input_path, output_path)

