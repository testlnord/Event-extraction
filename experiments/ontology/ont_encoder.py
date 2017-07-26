import logging as log
import re
from copy import copy
from itertools import chain

import numpy as np
import spacy
from intervaltree import IntervalTree, Interval

from experiments.tags import CategoricalTags
from experiments.ontology.symbols import POS_TAGS, DEP_TAGS


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
    ends, starts = zip(*matches) if len(matches) != 0 else ([], [])
    starts = [0] + list(starts)
    ends = list(ends) + [len(ctext)]
    spans = list(zip(starts, ends))
    itree = IntervalTree.from_tuples(spans)
    ssent = itree[crecord.s_startr:crecord.s_endr]
    if ssent == itree[crecord.o_startr:crecord.o_endr]:
        p = ssent.pop()
        cr = copy(crecord)
        cr.cut_context(p.begin, p.end)
        return cr


def expand_ents(token_seq, expand_noun_chunks=False):
    """
    Expand noun chunks and ents (add them with tokens to the new list) from underlying doc. Assuming all tokens are from the same doc
    :param token_seq: sequence of tokens (i.e. list)
    :return: list of tokens, including tokens from @token_seq
    """
    doc = token_seq[0].doc
    path = list(token_seq)
    inds = set(token.i for token in path)
    for span in (chain(doc.ents, doc.noun_chunks) if expand_noun_chunks else doc.ents):
        s_inds = set(t.i for t in span)
        if not inds.isdisjoint(s_inds):
            inds.update(s_inds)
    tokens = [doc[i] for i in sorted(inds)]
    return tokens


# todo: context subtrees around key spans?
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

    if include_spans:
        path = list(span1) + path + list(span2)

    # Sort to match the sentence order
    path = list(sorted(path, key=lambda token: token.i))

    # Extract context in the dep tree for border tokens in path
    left = path[0]
    lefts = list(left.lefts)
    lefts = lefts[max(0, len(lefts) - nb_context_tokens):]  # todo: duplicates on the left sometimes appear
    right = path[-1]
    rights = list(right.rights)
    lr = len(rights)
    rights = rights[:min(lr, nb_context_tokens)]
    return list(lefts) + path + list(rights)


def chars2spans(doc, *char_offsets_pairs):
    l = len(doc)-1
    lt = len(doc.text)+1
    charmap = IntervalTree(Interval(doc[i].idx, doc[i+1].idx, i) for i in range(l))
    charmap.addi(doc[-1].idx, lt, l)
    charmap.addi(lt, lt+1, l+1)  # as offsets_pairs are not-closed intervals, we need it for tokens at the end
    # tmp = [(t.idx, t.i) for t in doc]
    # charmap = dict([(i, ti) for k, (idx, ti) in enumerate(tmp[:-1]) for i in range(idx, tmp[k+1][0])])

    # def ii(p):
    #     i = charmap[p].pop()
    #     return i.data + int(p != i.begin)  # handle edge case when point falls into the previous interval
    # return [doc[ii(a):ii(b)] for a, b in char_offsets_pairs]
    def ii(p): return charmap[p].pop().data  # help function for convenience
    slices = []
    for a, b in char_offsets_pairs:
        ia = ii(a)
        ib = ii(b)
        ib += int(ib == ia)  # handle edge case when point falls into the previous interval which results in empty span
        slices.append(doc[ia:ib])
    return slices


def crecord2spans(crecord, nlp):
    return chars2spans(nlp(crecord.context), crecord.s_spanr, crecord.o_spanr)


class DBPediaEncoder:
    def __init__(self, nlp, superclass_map, inverse_map, expand_context=1, augment_data=False):
        self.nlp = nlp
        self.classes = superclass_map
        self.inverse_map = inverse_map
        self.tags = CategoricalTags(set(self.classes.values()))
        self.pos_tags = CategoricalTags(POS_TAGS)
        self.dep_tags = CategoricalTags(DEP_TAGS + [''])  # some punctuation marks can have no dependency tag
        self.channels = 3  # pos_tags, dep_tags, word_vectors
        self._expand_context = expand_context
        self.augment_data = augment_data

    @property
    def vector_length(self):
        return len(self.pos_tags), len(self.dep_tags), self.nlp.vocab.vectors_length

    @property
    def nbclasses(self):
        return len(self.tags)

    def __call__(self, crecords):
        for crecord in crecords:
            for xy in self.encode(crecord):
                yield xy

    def encode_data(self, s_span, o_span):
        """
        Encode data having entity spans (underlying doc is used implicitly by the means of dependency tree traversal)
        :param s_span: subject span
        :param o_span: object span
        :return: encoded data (tuple of arrays)
        """
        sdp = shortest_dep_path(s_span, o_span, include_spans=True, nb_context_tokens=self._expand_context)
        sdp = expand_ents(sdp)
        self.last_sdp = sdp  # if there're any need to look at that (for example, for testing)
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
            # dep = np.sum([np.array(dep_tags.encode(dep_var)) for dep_var in dep_vars])  # incorporate all dep types provided by dep parser...
            _dep_tags.append(dep)
        return np.array(_pos_tags), np.array(_dep_tags), vectors

    def encode_class(self, r):
        """
        Encode relation by numeric values.
        :param r: relation (of type: str)
        :return: one-hot vector of categories
        """
        raw_cls = self.classes.get(r)
        cls = self.tags.encode(raw_cls)
        return np.array(cls)

    def encode(self, crecord):
        s_span, o_span = crecord2spans(crecord, self.nlp)
        data = self.encode_data(s_span, o_span)
        cls = self.encode_class(crecord.r)
        yield (*data, cls)
        if self.augment_data:
            rr = self.inverse_map.get(crecord.r)
            if rr is not None:
                rcls = self.encode_class(rr)
                # todo: change data somewhow to reflect the change in directionality of the relation
                rdata = map(np.flipud, data)  # reverse all arrays
                # rdata = data
                yield (*rdata, rcls)

        # return (*self.encode_data(s_span, o_span), self.encode_class(crecord.r))
        # return (sdp, *self.encode_data(s_span, o_span), self.encode_class(crecord.r))  # for testing-looking


if __name__ == "__main__":
    from experiments.ontology.sub_ont import nlp
    from experiments.ontology.data import read_dataset, props_dir, load_superclass_mapping, load_inverse_mapping, load_all_data

    log.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=log.INFO)

    # scls_file = props_dir + 'prop_classes.csv'
    # inv_file = props_dir + 'prop_inverse.csv'
    sclasses = load_superclass_mapping()
    inverse = load_inverse_mapping()
    dataset = load_all_data(sclasses, shuffle=False)
    encoder = DBPediaEncoder(nlp, sclasses, inverse)

    contexts_dir = '/home/user/datasets/dbpedia/contexts/'
    filename = 'test4_{}.csv'.format('influenced')
    for i, cr in enumerate(dataset):
        for data in encoder.encode(cr):
            data, cls = data[:-1], data[-1]
            print()
            print(i, cr.triple)

            sdp = encoder.last_sdp
            s2 = expand_ents(sdp)
            s3 = expand_ents(sdp, True)
            print(sdp)
            if len(sdp) != len(s2):
                print(s2)
                print(s3)
            # for tok, pos, dep, vec in zip(sdp, *data):
            #     print(tok.text.ljust(20), tok.pos_.ljust(10), tok.dep_.ljust(10))

