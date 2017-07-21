import logging as log
import spacy

from experiments.ontology.symbols import POS_TAGS, DEP_TAGS
from experiments.ontology.sub_ont import read_dataset
from experiments.tags import CategoricalTags


def shortest_dep_path(span1, span2):
    """Find shortest path that connects two subtrees (span1 and span2) in the dependency tree"""
    path = []
    ancestors1 = list(span1.root.ancestors)
    for anc in span2.root.ancestors:
        path.append(anc)
        # If we find the nearest common ancestor
        if anc in ancestors1:
            # Add to common path subpath from span1 to common ancestor
            edge = ancestors1.index(anc)
            path.extend(ancestors1[:edge])
            break
    # Sort to match sentence order
    path = list(sorted(path, key=lambda token: token.i))
    return path

buf = {}
# todo: make some buffer for charmap; determine aricles (docs) by article_id
# todo: estimate size of full (ffiled) dict; make for any offset
def chars2spans(doc, *char_offsets_pairs):
    charmap = {t.idx: t.i for t in doc}
    charmap.update({t.idx+len(t): t.i+1 for t in doc})
    return [doc[charmap[a]:charmap[b]] for a, b in char_offsets_pairs]

nlp = spacy.load('en')
pos_tags = CategoricalTags(POS_TAGS)
dep_tags = CategoricalTags(DEP_TAGS)
# todo: do something with the interface?
def encode(s, r, o, s0, s1, o0, o1, ctext, cstart, cend, artid):
    doc = nlp(ctext)
    sent, s_span, o_span = chars2spans(doc, (cstart, cend), (s0, s1), (o0, o1))
    sdp = shortest_dep_path(s_span, o_span)
    vectors = []
    _pos_tags = []
    _dep_tags = []
    for t in sdp:
        log.info('token: {}; dep_: {}; pos_: {};'.format(t.text, t.dep_, t.pos_))
        vectors.append(t.vector)
        _pos_tags.append(pos_tags.encode(t.pos_))
        # _dep_tags.append(dep_tags.encode(t.dep_.lower()))
        dep_vars = t.dep_.lower().split('||')  # something very strange!!! need to get the data that makes dep parser do strange things
        # dep = sum([dep_tags.encode(dep_var) for dep_var in dep_vars])
        dep = dep_tags.encode(dep_vars[0])
        _dep_tags.append(dep)
    return sdp, _pos_tags, _dep_tags, vectors


if __name__ == "__main__":
    log.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=log.DEBUG)

    contexts_dir = '/home/user/datasets/dbpedia/contexts/'
    filename = 'test3_{}.csv_'.format('computingPlatform')
    print('Starting...')
    for data in read_dataset(contexts_dir + filename):
        for tok, pos, dep, vec in zip(*encode(*data)):
            print(tok.text.ljust(40), tok.pos_.ljust(20), tok.dep_.ljust(20), len(pos), len(dep), len(vec))



