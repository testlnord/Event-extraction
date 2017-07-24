import logging as log
import csv
import re
from copy import copy

import spacy
from intervaltree import IntervalTree, Interval

from experiments.tags import CategoricalTags
from experiments.ontology.symbols import POS_TAGS, DEP_TAGS
from experiments.ontology.sub_ont import read_dataset


def is_connected(span):
    r = span.root
    return (r.left_edge.i == 0) and (r.right_edge.i == len(span)-1)
    # return len(r.subtree) == len(span.subtree)  # another way


def filter_context(crecord):
    """Only chooses the sentence where the entities (subject and object) are present.
    Does not yield other sentences. Returns original crecord if it is valid."""
    ctext = crecord.context
    rex = '\n+'
    matches = [m.span() for m in re.finditer(rex, ctext)]
    ends, starts = list(zip(*matches))
    starts = [0] + list(starts)
    ends = list(ends) + [len(ctext)]
    spans = list(zip(starts, ends))
    itree = IntervalTree.from_tuples(spans)
    ssent = itree[crecord.s_startr:crecord.s_endr]
    if ssent == itree[crecord.o_startr:crecord.o_endr]:
        p = ssent.pop()
        cr = copy(crecord)
        cr.context = ctext[p.begin:p.end]
        cr.cstart = crecord.cstart + p.begin
        cr.cend = crecord.cstart + p.end
        return cr


# todo: possible need to preserve noun_chunks (and named entities -- or remove them...)
def shortest_dep_path(span1, span2, include_spans=True, nb_context_tokens=0):
    """
    Find shortest path that connects two subtrees (span1 and span2) in the dependency tree.
    :param span1: one of the spans
    :param span2: one of the spans
    :param include_spans: whether include source spans in the SDP or not
    :param nb_context_tokens: number of tokens for expanding SDP to the right and to the left (default 0, i.e. plain SDP)
    :return: list of tokens
    """
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
    if include_spans:
        path = list(span1) + path + list(span2)
    # Extract context in the dep tree for border tokens in path
    left = path[0]
    lefts = list(left.lefts)
    lefts = lefts[max(0, len(lefts) - nb_context_tokens):]
    right = path[-1]
    rights = list(right.rights)
    lr = len(rights)
    rights = rights[:min(lr, nb_context_tokens)]
    return list(lefts) + path + list(rights)


# context around key spans (not just sdp), maybe subtrees?
# make some buffer for charmap; determine aricles (docs) by article_id

def chars2spans(doc, *char_offsets_pairs):
    l = len(doc)-1
    charmap = IntervalTree(Interval(doc[i].idx, doc[i+1].idx, i) for i in range(l))
    charmap.addi(doc[-1].idx, len(doc.text)+1, l)
    # tmp = [(t.idx, t.i) for t in doc]
    # charmap = dict([(i, ti) for k, (idx, ti) in enumerate(tmp[:-1]) for i in range(idx, tmp[k+1][0])])
    def ii(p): return charmap[p].pop().data
    return [doc[ii(a):ii(b)] for a, b in char_offsets_pairs]


class DBPediaEncoder:
    def __init__(self, nlp, rel_classes_dict, expand_context=1, charmap_buf_size=4):
        self.nlp = nlp
        self.classes = rel_classes_dict
        self.output_tags = CategoricalTags(list(self.classes.keys()))
        self.pos_tags = CategoricalTags(POS_TAGS)
        self.dep_tags = CategoricalTags(DEP_TAGS)
        self.channels = 3  # pos_tags, dep_tags, word_vectors
        self._expand_context = expand_context
        self._charmaps = {}
        self._maxbuf = charmap_buf_size  # for chars2spans

    @property
    def nbclasses(self):
        return len(self.pos_tags), len(self.dep_tags), self.nlp.vocab.vector_length

    # todo: lookup by article_id is wrong! docs are not articles, they're sentences from articles
    def chars2spans(self, doc, article_id, *char_offsets_pairs):
        if article_id not in self._charmaps:
            l = len(doc)-1
            charmap = IntervalTree(Interval(doc[i].idx, doc[i+1].idx, i) for i in range(l))
            charmap.addi(doc[-1].idx, len(doc.text)+1, l)
            if len(self._charmaps) >= self._maxbuf:
                self._charmaps.popitem()
            self._charmaps[article_id] = charmap
        def ii(p):
            return self._charmaps[article_id].search(p).pop().data
        return [doc[ii(a):ii(b)] for a, b in char_offsets_pairs]

    def __call__(self, crecords):
        for crecord in crecords:
            yield self.encode(crecord)

    def encode(self, crecord):
        sent = nlp(crecord.context)
        s_span, o_span = chars2spans(sent, crecord.s_spanr, crecord.o_spanr)
        sdp = shortest_dep_path(s_span, o_span, include_spans=True, nb_context_tokens=self._expand_context)
        vectors = []
        _pos_tags = []
        _dep_tags = []
        for t in sdp:
            log.debug('token: {}; dep_: {}; pos_: {};'.format(t.text, t.dep_, t.pos_))
            vectors.append(t.vector)
            _pos_tags.append(self.pos_tags.encode(t.pos_))
            # _dep_tags.append(dep_tags.encode(t.dep_))
            dep_vars = t.dep_.split('||')  # something very strange!!! need to get the data that makes dep parser do strange things
            dep = self.dep_tags.encode(dep_vars[0])
            # dep = sum([dep_tags.encode(dep_var) for dep_var in dep_vars])  # incorporate all dep types provided by dep parser...
            _dep_tags.append(dep)

        raw_cls = self.classes.get(crecord.r) / 2  # ignoring direction of the relation. for now.
        cls = self.output_tags.encode(raw_cls)
        # return (_pos_tags, _dep_tags, vectors), cls
        return (sdp, _pos_tags, _dep_tags, vectors), raw_cls  # for testing-looking


props_dir='/home/user/datasets/dbpedia/qs/props/'
def load_classes_dict(filename):
    """Mapping between relations and final classes"""
    classes = {}
    with open(filename, 'r', newline='') as f:
        reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC, quotechar=' ')
        for cls, rel, _ in reader:
            if int(cls) >= 0:
                classes[rel] = int(cls)
    return classes


import numpy as np
import os
contexts_dir = '/home/user/datasets/dbpedia/contexts/'
def load_all_data(classes, data_dir=contexts_dir):
    """Load ContextRecords with classes from @classes from all files in the directory @data_dir and shuffle it."""
    dataset = []
    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        if os.path.isfile(filepath):
            for crecord in read_dataset(filepath):
                if crecord.relation in classes:
                    filtered = filter_context(crecord)
                    if filtered is not None:
                        dataset.append(filtered)
    np.random.shuffle(dataset)
    return dataset


if __name__ == "__main__":
    from experiments.ontology.sub_ont import nlp
    log.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=log.INFO)

    classes = load_classes_dict()
    encoder = DBPediaEncoder(nlp, classes)

    contexts_dir = '/home/user/datasets/dbpedia/contexts/'
    filename = 'test3_{}.csv_'.format('computingPlatform')
    print('Starting...')
    for i, cr in enumerate(read_dataset(contexts_dir + filename)):
        data, cls = encoder.encode(cr)
        print()
        print(i, cls, cr.triple)
        for tok, pos, dep, vec in zip(*data):
            print(tok.text.ljust(20), tok.pos_.ljust(10), tok.dep_.ljust(10))

