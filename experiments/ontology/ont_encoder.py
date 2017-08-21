import logging as log
from itertools import product

import numpy as np

from experiments.tags import CategoricalTags
from experiments.ontology.symbols import POS_TAGS, DEP_TAGS, IOB_TAGS, NER_TAGS
from experiments.nlp_utils import *


def crecord2spans(crecord, nlp):
    return chars2spans(nlp(crecord.context), crecord.s_spanr, crecord.o_spanr)


class DBPediaEncoder:
    def __init__(self, nlp, superclass_map, inverse_relations=None,
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
        self._inverse_map = dict()
        assert isinstance(superclass_map, dict)

        # All relations not in classes.values() (that is, unknown relations) will be tagged with the default_tag (i.e. examples of no-relations)
        raw_classes = list(product(sorted(set(self.classes.values())), [True, False]))
        if inverse_relations:
            for rx, ry in inverse_relations.items():
                # Map each class to its' inverse (with changed direction, of course)
                #   if at least its' inverse is in the domain (in raw_classes)
                x, y = (rx, False), (ry, True)
                if y in raw_classes:
                    self._inverse_map[x] = y
                    log.info('DBPediaEncoder: map class to inverse: {}->{}'.format(x, y))
                if x in raw_classes:
                    raw_classes.remove(x)
        log.info('DBPediaEncoder: classes: {}'.format(raw_classes))
        self.tags = CategoricalTags(raw_classes, default_tag=(None, None))

        self.iob_tags = CategoricalTags(sorted(IOB_TAGS))
        self.ner_tags = CategoricalTags(sorted(NER_TAGS), default_tag='')  # default_tag for non-named entities
        self.pos_tags = CategoricalTags(sorted(POS_TAGS))
        self.dep_tags = CategoricalTags(sorted(DEP_TAGS), default_tag='')  # in case of unknown dep tags (i.e. some punctuation marks can have no dependency tag)
        self._expand_context = expand_context
        self.expand_noun_chunks = expand_noun_chunks

    @property
    def channels(self):
        return len(self.vector_length)

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
        sdp, self.last_sdp_root = shortest_dep_path(s_span, o_span, include_spans=True, nb_context_tokens=self._expand_context)
        sdp = expand_ents(sdp, self.expand_noun_chunks)
        self.last_sdp = sdp  # for the case of any need to look at that (as example, for testing)
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

    def encode_raw_class(self, crecord):
        rel_cls = self.classes.get(str(crecord.relation))
        # NB: raw_cls may be None (filtered by self.classes), but still have specified direction, so we get (None, True) or (None, False)
        raw_cls = (rel_cls, crecord.direction)
        raw_cls = self._inverse_map.get(raw_cls, raw_cls)
        return raw_cls

    def encode_class(self, crecord):
        """
        Encode relation by numeric values.
        :param r: relation (of type: str)
        :return: tuple of (one-hot vector of categories, direction of the s->o relation)
        """
        cls = self.tags.encode(self.encode_raw_class(crecord))
        return (np.array(cls),)  # packing to tuple to be consistent with unpacking in encode() todo: remove

    def encode(self, crecord):
        s_span, o_span = crecord2spans(crecord, self.nlp)
        data = self.encode_data(s_span, o_span)
        cls = self.encode_class(crecord)
        yield (*data, *cls)


class DBPediaEncoderWithEntTypes(DBPediaEncoder):
    from experiments.ontology.linker import NERTypeResolver
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

    # todo: linking with existing uri should happen here
    # def encode_data(self, s_span, o_span):
    #     data = super().encode_data(s_span, o_span)
    #     uri = None
    #     s_type = self._ent_tags.encode(self._ner_type_resolver.get_by_uri(uri))
    #     o_type = self._ent_tags.encode(self._ner_type_resolver.get_by_uri(uri))

    def encode(self, crecord):
        s_span, o_span = crecord2spans(crecord, self.nlp)

        # Set the entities types on crecord's doc here as ground truth
        #   and call data.encode_data() which will use these anno
        data = super().encode_data(s_span, o_span)
        cls = self.encode_class(crecord)

        # Do not use data linking made in overloaded encode_data (hence, super()), use info from crecord instead
        s_type = self._encode_type(crecord.subject)
        o_type = self._encode_type(crecord.object)
        len_output = len(self.last_sdp)

        # Feed entity types only on the corresponding to entity spans time steps
        s_type_out = [np.zeros_like(s_type)] * len_output; s_indices = {token.i for token in s_span}
        o_type_out = [np.zeros_like(o_type)] * len_output; o_indices = {token.i for token in o_span}
        for i, token in enumerate(self.last_sdp):
            if token.i in s_indices: s_type_out[i] = s_type
            if token.i in o_indices: o_type_out[i] = o_type

        # s_type_out = [s_type] * len_output  # feeding type on each timestep
        # o_type_out = [o_type] * len_output  # feeding type on each timestep
        yield (*data, np.array(s_type_out), np.array(o_type_out), *cls)


class DBPediaEncoderBranched(DBPediaEncoder):
    @property
    def vector_length(self):
        _vl = super().vector_length
        return (*_vl, *_vl)

    def encode_data(self, s_span, o_span):
        data = super().encode_data(s_span, o_span)
        i = self.last_sdp_root
        left_part = [d[:i+1] for d in data]
        right_part = [d[i:] for d in data]
        return left_part + right_part


# deprecated todo: change or remove
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


if __name__ == "__main__":
    import os
    import spacy
    from experiments.ontology.tagger import load_golden_data
    from experiments.ontology.data import load_rc_data, filter_context
    from experiments.ontology.data_structs import RelationRecord, RelRecord
    from experiments.data_utils import unpickle
    from experiments.ontology.symbols import RC_CLASSES_MAP, RC_CLASSES_MAP_ALL, RC_INVERSE_MAP
    from experiments.ontology.config import config

    # Pipeline:
    # doc = nlp(text)
    # linked_doc = link_ents(doc)
    # rel_candidates = generate_candidates(linked_doc)
    #   there could be some already present in the ontology candidates...
    # rel_preds = model.predict(rel_candidates)
    #   filter unsure ones

    # load nlp model (our?)
    #   ! align with symbols classes
    #       ? some test of nlp classes? or somehow get them...
    # inst encoder
    #   load classes from symbols
    #   ! link linker to encoder
    #       load linker model
    #   ner type resolver
    # inst net, load it's model

    # global things:
    #   graphs & namespaces
    #   nlp
    #   linker, type resolver

    log.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=log.INFO)

    sclasses = RC_CLASSES_MAP_ALL
    inverse = RC_INVERSE_MAP
    nlp = spacy.load(**config['models']['nlp'])
    encoder = DBPediaEncoderWithEntTypes(nlp, sclasses, inverse_relations=inverse)

    data_dir = config['data']['dir']
    rc_out = os.path.join(data_dir, 'rc', 'rrecords.v2.filtered.pck')
    rc0_out = os.path.join(data_dir, 'rc', 'rrecords.v2.negative.pck')
    dataset = load_rc_data(sclasses, rc_out, rc0_out, neg_ratio=0., shuffle=False)
    print('total with filtered classes:', len(dataset))

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
        text = record.context.strip()
        xs = record.s_startr
        s = text[xs:record.s_endr]
        xo = record.o_startr
        o = text[xo:record.o_endr]
        print('t:', text)
        print('s:', s, 'true:', str(record.subject))
        print('o:', o, 'true:', str(record.object))
