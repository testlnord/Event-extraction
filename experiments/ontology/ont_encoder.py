import logging as log
from itertools import product
from collections import defaultdict

import numpy as np

from experiments.tags import CategoricalTags
from experiments.ontology.symbols import POS_TAGS, DEP_TAGS, IOB_TAGS, NER_TAGS
from experiments.nlp_utils import *


def transform_lists(text):
    rex = '^- +'  # todo: why it doesn't work?
    # group all subsequent '-' lines; make from them comma-list
    # using previous Section name or previous sentence (especially if it ends on colon)


def crecord2spans(crecord, nlp):
    return chars2spans(nlp(crecord.context), crecord.s_spanr, crecord.o_spanr)


# Reminder: to change the number of channels, change: self.channels, self.vector_length, self.encode_data


class DBPediaEncoder:
    def __init__(self, nlp, superclass_map,
                 expand_context=1, expand_noun_chunks=False):
        """
        Encodes the data into the suitable for the Keras format. Uses Shortest Dependency Tree (SDP) assumption.
        :param nlp:
        :param superclass_map: mapping of base relation types (uris) into final relation types used for classification
        :param expand_context: expand context around key tokens in SDP
        :param expand_noun_chunks: whether to expand parts of noun chunks in SDP into full noun chunks
        """
        self.nlp = nlp
        self.classes = superclass_map
        # All relations not in classes.values() (that is, unknown relations) will be tagged with the default_tag (i.e. examples of no-relations)
        # self.tags = CategoricalTags(set(self.classes.values()), default_tag='')
        # self.direction_tags = CategoricalTags({None, True, False})  # None for negative relations (they have no direction)
        raw_classes = list(product(sorted(set(self.classes.values())), [True, False]))
        log.info('DBPediaEncoder: classes: {}'.format(raw_classes))
        self.tags = CategoricalTags(raw_classes, default_tag='-')
        self.iob_tags = CategoricalTags(sorted(IOB_TAGS))
        self.ner_tags = CategoricalTags(sorted(NER_TAGS), default_tag='')  # default_tag for non-named entities
        self.pos_tags = CategoricalTags(sorted(POS_TAGS))
        self.dep_tags = CategoricalTags(sorted(DEP_TAGS), default_tag='')  # in case of unknown dep tags (i.e. some punctuation marks can have no dependency tag)
        self._expand_context = expand_context
        self.expand_noun_chunks = expand_noun_chunks

    @property
    def channels(self):
        return 5  # iob_tags, ner_tags, pos_tags, dep_tags, word_vectors

    @property
    def vector_length(self):
        wv = self.wordvec_length
        return len(self.iob_tags), len(self.ner_tags), len(self.pos_tags), len(self.dep_tags), wv

    @property
    def wordvec_length(self):
        return self.nlp.vocab.vectors_length

    @property
    def nbclasses(self):
        return len(self.tags)

    def __call__(self, crecords):
        for crecord in crecords:
            yield from self.encode(crecord)

    def encode_data(self, s_span, o_span):
        """
        Encode data having entity spans (underlying doc is used implicitly by the means of dependency tree traversal)
        :param s_span: subject span
        :param o_span: object span
        :return: encoded data (tuple of arrays)
        """
        sdp, iroot = shortest_dep_path(s_span, o_span, include_spans=True, nb_context_tokens=self._expand_context)
        sdp = expand_ents(sdp, self.expand_noun_chunks)
        self.last_sdp = sdp  # for the case of any need to look at that (as example, for testing)
        self.last_sdp_root = iroot
        _iob_tags = []
        _ner_tags = []
        _pos_tags = []
        _dep_tags = []
        _vectors = []
        _wn_hypernyms = []
        for t in sdp:
            log.debug('token: {}; ent_type_: {}; dep_: {}; pos_: {};'.format(t.text, t.ent_type_, t.dep_, t.pos_))
            _iob_tags.append(self.iob_tags.encode(t.ent_iob_))
            _ner_tags.append(self.ner_tags.encode(t.ent_type_))
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
            # hyp = get_hypernym(self.nlp, t)
            # _wn_hypernyms.append(np.zeros(self.wordvec_length) if hyp is None else hyp.vector)
        # data = _iob_tags, _ner_tags, _pos_tags, _dep_tags, _vectors, _wn_hypernyms
        data = _iob_tags, _ner_tags, _pos_tags, _dep_tags, _vectors
        return tuple(map(np.array, data))

    def encode_class(self, crecord):
        """
        Encode relation by numeric values.
        :param r: relation (of type: str)
        :return: tuple of (one-hot vector of categories, direction of the s->o relation)
        """
        raw_cls = self.classes.get(str(crecord.r))
        # NB: raw_cls may be None (filtered by self.classes), but still have specified direction, so we get (None, True) or (None, False)
        cls = self.tags.encode((raw_cls, crecord.direction))
        return (np.array(cls),)  # packing to tuple to be consistent with unpacking in encode()
        # direction = self.direction_tags.encode(crecord.direction)
        # return np.array(cls), direction

    def encode(self, crecord):
        s_span, o_span = crecord2spans(crecord, self.nlp)
        data = self.encode_data(s_span, o_span)
        cls = self.encode_class(crecord)
        yield (*data, *cls)


class DBPediaEncoderWithEntTypes(DBPediaEncoder):
    from experiments.ontology.sub_ont import NERTypeResolver
    from experiments.ontology.symbols import ENT_CLASSES
    _ner_type_resolver = NERTypeResolver()
    _ent_tags = CategoricalTags(sorted(ENT_CLASSES), default_tag='')

    def _encode_type(self, uri):
        return self._ent_tags.encode(self._ner_type_resolver.get_by_uri(uri))

    @property
    def vector_length(self):
        _vl = super().vector_length
        _l = len(self._ent_tags)
        return (*_vl, _l, _l)

    @property
    def channels(self):
        return super().channels + 2

    # todo: linking with existing uri should happen here
    # def encode_data(self, s_span, o_span):
    #     data = super().encode_data(s_span, o_span)
    #     uri = None
    #     s_type = self._ent_tags.encode(self._ner_type_resolver.get_by_uri(uri))
    #     o_type = self._ent_tags.encode(self._ner_type_resolver.get_by_uri(uri))

    def encode(self, crecord):
        s_span, o_span = crecord2spans(crecord, self.nlp)
        # Do not use data linking made in overloaded encode_data (hence, super()), use info from crecord instead
        s_type = self._encode_type(crecord.subject)
        o_type = self._encode_type(crecord.object)

        # Set the entities types on crecord's doc here as ground truth
        #   and call data.encode_data() which will use these anno
        data = super().encode_data(s_span, o_span)
        cls = self.encode_class(crecord)

        # todo: think about feeding on each timestep
        s_type = [s_type] * len(self.last_sdp)  # feeding type on each timestep
        o_type = [o_type] * len(self.last_sdp)  # feeding type on each timestep
        yield (*data, np.array(s_type), np.array(o_type), *cls)


class DBPediaEncoderBranched(DBPediaEncoder):
    @property
    def vector_length(self):
        _vl = super().vector_length
        return (*_vl, *_vl)

    @property
    def channels(self):
        return super().channels * 2

    def encode_data(self, s_span, o_span):
        data = super().encode_data(s_span, o_span)
        i = self.last_sdp_root
        left_part = [d[:i+1] for d in data]
        right_part = [d[i:] for d in data]
        return left_part + right_part


class EncoderDataAugmenter:
    """Wrapper of DBPediaEncoder for data augmentation."""
    def __init__(self, encoder, inverse_map):
        """
        :param encoder: DBPediaEncoder or its' subclass
        :param inverse_map: mapping of final relation types into their inverse relation types (used for data augmentation)
        """
        self.inverse_map = inverse_map
        self.encoder = encoder

    def encode(self, crecord):
        e = self.encoder
        s_span, o_span = crecord2spans(crecord, e.nlp)
        data = e.encode_data(s_span, o_span)
        cls = e.encode_class(crecord)
        yield (*data, *cls)

        rr = self.inverse_map.get(crecord.r)
        crecord.r = rr
        if rr is not None:
            rcls = e.encode_class(crecord)
            # todo: change data somewhow to reflect the change in the directionality of relation
            rdata = map(np.flipud, data)  # reverse all arrays
            yield (*rdata, *rcls)


def sort_simmetric(dataset):
    """Sort dataset by relation and the directionality of it."""
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
    import os
    from experiments.ontology.data import nlp, load_rc_data, filter_context
    from experiments.ontology.data_structs import RelationRecord
    from experiments.data_utils import unpickle
    from experiments.ontology.symbols import RC_CLASSES_MAP

    log.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=log.INFO)

    sclasses = RC_CLASSES_MAP
    raw_classes1 = list(product(sorted(set(sclasses.values())), [True, False]))
    raw_classes2 = list(product(sorted(set(sclasses.values())), [True, False]))
    raw_classes3 = list(product(sorted(set(sclasses.values())), [True, False]))
    assert all(c1 == c2 == c3 for c1, c2, c3 in zip(raw_classes1, raw_classes2, raw_classes3))
    exit(0)


    data_dir = '/home/user/datasets/dbpedia/'
    rc_out = os.path.join(data_dir, 'rc', 'rrecords.v2.filtered.pck')
    rc0_out = os.path.join(data_dir, 'rc', 'rrecords.v2.negative.pck')
    dataset = load_rc_data(sclasses, rc_out, rc0_out, neg_ratio=0., shuffle=False)
    print('total with filtered classes:', len(dataset))

    encoder = DBPediaEncoderWithEntTypes(nlp, sclasses)
    c = encoder.channels
    assert len(encoder.vector_length) == encoder.channels

    # find_simmetric(dataset)
    # exit()

    dataset = list(unpickle(rc_out))
    _valid = [rr for rr in dataset if rr.valid_offsets]
    _fc = list(filter(None, map(filter_context, _valid)))
    print('TOTAL:', len(dataset))
    print('VALID:', len(_valid))
    print('(BAD:', len(dataset) - len(_valid), ')')
    print('VALID FILTERED:', len(_fc))
    input('press enter to proceed...')

    bad = 0
    for i, record in enumerate(_fc):
        print()
        # final_rel = sclasses.get(str(record.relation))
        # true_tag = (final_rel, record.direction)

        text = record.context.strip()
        xs = record.s_startr
        s = text[xs:record.s_endr]
        xo = record.o_startr
        o = text[xo:record.o_endr]
        print('t:', text)
        print('s:', s, 'true:', str(record.subject))
        print('o:', o, 'true:', str(record.object))

        # for data in encoder.encode(cr):
        #     data, clss = data[:c], data[c:]
        #     print()
        #     print(i, cr.triple)
        #
        #     sdp = encoder.last_sdp
        #     s2 = expand_ents(sdp)
        #     s3 = expand_ents(sdp, True)
        #     print(sdp)
        #     if len(sdp) != len(s2):
        #         print(s2)
        #         print(s3)
        #     for tok, iob, pos, dep, vec in zip(sdp, *data):
        #         print(tok.text.ljust(20), tok.pos_.ljust(10), tok.dep_.ljust(10))
