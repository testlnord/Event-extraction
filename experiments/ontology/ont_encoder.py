import logging as log
import spacy
from intervaltree import IntervalTree, Interval

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


# todo: make some buffer for charmap; determine aricles (docs) by article_id
def chars2spans(doc, *char_offsets_pairs):
    l = len(doc)-1
    charmap = IntervalTree(Interval(doc[i].idx, doc[i+1].idx, i) for i in range(l))
    charmap.addi(doc[-1].idx, len(doc.text)+1, l)
    def ii(p): return charmap[p].pop().data
    # tmp = [(t.idx, t.i) for t in doc]
    # charmap = dict([(i, ti) for k, (idx, ti) in enumerate(tmp[:-1]) for i in range(idx, tmp[k+1][0])])
    return [doc[ii(a):ii(b)] for a, b in char_offsets_pairs]


# nlp = spacy.load('en')
from experiments.ontology.sub_ont import nlp
pos_tags = CategoricalTags(POS_TAGS)
dep_tags = CategoricalTags(DEP_TAGS)
# todo: do something with the interface?
def encode(s, r, o, s0, s1, o0, o1, ctext, cstart, cend, artid):
    sent = nlp(ctext)
    s = (s0-cstart, s1-cstart)  # offsets in dataset are relative to the whole document, not sentence
    o = (o0-cstart, o1-cstart)
    s_span, o_span = chars2spans(sent, s, o)
    sdp = shortest_dep_path(s_span, o_span)
    vectors = []
    _pos_tags = []
    _dep_tags = []
    for t in sdp:
        log.info('token: {}; dep_: {}; pos_: {};'.format(t.text, t.dep_, t.pos_))
        vectors.append(t.vector)
        _pos_tags.append(pos_tags.encode(t.pos_))
        # _dep_tags.append(dep_tags.encode(t.dep_))
        dep_vars = t.dep_.split('||')  # something very strange!!! need to get the data that makes dep parser do strange things
        dep = dep_tags.encode(dep_vars[0])
        # dep = sum([dep_tags.encode(dep_var) for dep_var in dep_vars])
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



