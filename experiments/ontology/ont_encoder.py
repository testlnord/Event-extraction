import logging as log
import re
from copy import copy

import numpy as np
import spacy

from experiments.tags import CategoricalTags
from experiments.ontology.symbols import POS_TAGS, DEP_TAGS, IOB_TAGS, NER_TAGS
from experiments.nlp_utils import *


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


def crecord2spans(crecord, nlp):
    return chars2spans(nlp(crecord.context), crecord.s_spanr, crecord.o_spanr)


# Reminder: to change the number of channels, change: self.channels, self.vector_length, self.encode_data

class DBPediaEncoder:
    def __init__(self, nlp, superclass_map, inverse_map,
                 expand_context=1, expand_noun_chunks=False, augment_data=False):
        self.nlp = nlp
        self.classes = superclass_map
        self.inverse_map = inverse_map
        self.tags = CategoricalTags(set(self.classes.values()))
        self.iob_tags = CategoricalTags(IOB_TAGS)
        self.pos_tags = CategoricalTags(POS_TAGS)
        self.dep_tags = CategoricalTags(DEP_TAGS + [''])  # some punctuation marks can have no dependency tag
        # self.channels = 4 + int(add_wordnet)  # iob_tags, pos_tags, dep_tags, word_vectors, wordnet hypernyms vectors
        self._expand_context = expand_context
        self.expand_noun_chunks = expand_noun_chunks
        self.augment_data = augment_data

    @property
    def vector_length(self):
        wv = self.wordvec_length
        return len(self.iob_tags), len(self.pos_tags), len(self.dep_tags), wv

    @property
    def wordvec_length(self):
        return self.nlp.vocab.vectors_length

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
        sdp = expand_ents(sdp, self.expand_noun_chunks)
        self.last_sdp = sdp  # for the case of any need to look at that (as example, for testing)
        _iob_tags = []
        _pos_tags = []
        _dep_tags = []
        _vectors = []
        _wn_hypernyms = []
        for t in sdp:
            log.debug('token: {}; dep_: {}; pos_: {};'.format(t.text, t.dep_, t.pos_))
            _iob_tags.append(self.iob_tags.encode(t.ent_iob_))
            _pos_tags.append(self.pos_tags.encode(t.pos_))
            _vectors.append(t.vector)
            # Dependency tags by spacy
            dep_vars = t.dep_.split('||')
            if len(dep_vars) > 1:
                log.info('DBPediaEncoder: dep_tags: strange dep: token:"{}" dep:"{}"'.format(t.text, t.dep_))
            # dep = self.dep_tags.encode(dep_vars[0])
            # Use all dep types provided by dep parser...
            dep = sum(np.array(self.dep_tags.encode(dep_var)) for dep_var in dep_vars)
            _dep_tags.append(dep)
            # WordNet hypernyms' word vectors
            hyp = get_hypernym(self.nlp, t)
            _wn_hypernyms.append(np.zeros(self.wordvec_length) if hyp is None else hyp.vector)
        # data = _iob_tags, _pos_tags, _dep_tags, _vectors, _wn_hypernyms
        data = _iob_tags, _pos_tags, _dep_tags, _vectors
        return tuple(map(np.array, data))

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
                yield (*rdata, rcls)


from collections import defaultdict
def sort_simmetric(dataset):
    direction_sorted = defaultdict(lambda : defaultdict(list))
    for cr in dataset:
        direction = (cr.s_end <= cr.o_start)
        direction_sorted[cr.relation][direction].append(cr)
    return direction_sorted


def find_simmetric(dataset):
    for rel, d in sort_simmetric(dataset).items():
        print()
        lt = len(d[True])
        lf = len(d[False])
        print('rev: {}; norm: {};'.format(lf, lt), rel)
        if lt != 0 and lf != 0:
            for direction_same in d.keys():
                for cr in d[direction_same]:
                    print(int(direction_same), cr.triple)
                    print(' ', cr.context)



if __name__ == "__main__":
    from experiments.ontology.sub_ont import nlp
    from experiments.ontology.data import props_dir, load_superclass_mapping, load_inverse_mapping, load_all_data

    log.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=log.INFO)

    # scls_file = props_dir + 'prop_classes.csv'
    # inv_file = props_dir + 'prop_inverse.csv'
    sclasses = load_superclass_mapping()
    inverse = load_inverse_mapping()
    dataset = load_all_data(sclasses, shuffle=False)
    encoder = DBPediaEncoder(nlp, sclasses, inverse)

    find_simmetric(dataset)
    exit()

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
            # for tok, iob, pos, dep, vec in zip(sdp, *data):
            #     print(tok.text.ljust(20), tok.pos_.ljust(10), tok.dep_.ljust(10))

