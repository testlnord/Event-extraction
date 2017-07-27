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
        self.pos_tags = CategoricalTags(POS_TAGS)
        self.dep_tags = CategoricalTags(DEP_TAGS + [''])  # some punctuation marks can have no dependency tag
        self.iob_tags = CategoricalTags(IOB_TAGS)
        self.channels = 4  # pos_tags, dep_tags, word_vectors
        self._expand_context = expand_context
        self.expand_noun_chunks = expand_noun_chunks
        self.augment_data = augment_data

    @property
    def vector_length(self):
        return len(self.iob_tags), len(self.pos_tags), len(self.dep_tags), self.nlp.vocab.vectors_length

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
        _vectors = []
        _pos_tags = []
        _dep_tags = []
        _iob_tags = []
        for t in sdp:
            log.debug('token: {}; dep_: {}; pos_: {};'.format(t.text, t.dep_, t.pos_))
            _vectors.append(t.vector)
            _pos_tags.append(self.pos_tags.encode(t.pos_))
            _iob_tags.append(self.iob_tags.encode(t.ent_iob_))
            # _dep_tags.append(dep_tags.encode(t.dep_))
            dep_vars = t.dep_.split('||')  # something very strange!!! need to get the data that makes dep parser do strange things
            dep = self.dep_tags.encode(dep_vars[0])
            # dep = np.sum([np.array(dep_tags.encode(dep_var)) for dep_var in dep_vars])  # incorporate all dep types provided by dep parser...
            _dep_tags.append(dep)
        return np.array(_iob_tags), np.array(_pos_tags), np.array(_dep_tags), _vectors

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
            # for tok, iob, pos, dep, vec in zip(sdp, *data):
            #     print(tok.text.ljust(20), tok.pos_.ljust(10), tok.dep_.ljust(10))

