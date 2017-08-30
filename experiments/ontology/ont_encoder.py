import logging as log
from itertools import product

import numpy as np
from spacy.tokens import Span

from experiments.tags import CategoricalTags
from experiments.ontology.symbols import POS_TAGS, DEP_TAGS, IOB_TAGS, NER_TAGS, ALL_ENT_CLASSES, WORDNET_HYPERNYM_CLASSES
from experiments.ontology.linker import NERTypeResolver
from experiments.nlp_utils import *


def crecord2spans_old(crecord, nlp):
    return chars2spans(nlp(crecord.context), crecord.s_spanr, crecord.o_spanr)


def crecord2spans(cr, nlp, ntr):
    """
    Get token offsets using char offsets and merge entity annotations in Doc with spacy's using ner_type_resolver (ntr)
    :param cr: RelationRecord
    :param nlp: spacy model
    :param ntr: NERTypeResolver
    :return: Tuple[spacy.token.Span, spacy.token.Span]
    """
    doc = nlp(cr.context)
    spans = chars2spans(doc, cr.s_spanr, cr.o_spanr)
    uris = [cr.subject, cr.object]

    _v_ = nlp.vocab.strings
    true_ents = []
    for span, uri in zip(spans, uris):
        label_str = ntr.get_by_uri(uri, default_type=span.label_)  # fallback to spacy's ner tag
        if not label_str in _v_:
            log.info('crecord2spans: unknown entity type: "{}"'.format(label_str))
        label_id = _v_[label_str]  # modify StringStore if necessary
        true_ents.append(Span(doc=doc, start=span.start, end=span.end, label=label_id))
    corrected_ents = merge_ents_offsets(true_ents, doc.ents)
    doc.ents = corrected_ents
    return true_ents


class DBPediaEncoder:
    def __init__(self, nlp, superclass_map, inverse_relations=None,
                 min_entities_dist=0, expand_context=0,
                 expand_noun_chunks=False, ner_type_resolver=NERTypeResolver()):
        """
        Encodes the data into the suitable for the Keras format. Uses Shortest Dependency Tree (SDP) assumption.
        :param nlp:
        :param superclass_map: mapping of base relation types (uris) into final relation types used for classification
        :param inverse_relations: mapping between relations and their inverse (symmetric) relations
        :param min_entities_dist: minimum distance between entities, otherwise add context (#expand_context tokens) around them to SDP
        :param expand_context: expand context around key tokens in SDP by that number of tokens
        :param expand_noun_chunks: whether to expand parts of noun chunks in SDP into full noun chunks
        """
        self.ntr = ner_type_resolver
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
        self.ner_tags = CategoricalTags(sorted(ALL_ENT_CLASSES), default_tag='')  # default_tag for non-named entities
        self.pos_tags = CategoricalTags(sorted(POS_TAGS))
        self.dep_tags = CategoricalTags(sorted(DEP_TAGS), default_tag='')  # in case of unknown dep tags (i.e. some punctuation marks can have no dependency tag)
        self.wn_tags = CategoricalTags(sorted(WORDNET_HYPERNYM_CLASSES), default_tag='')

        self._min_entities_dist = min_entities_dist
        self._expand_context = expand_context
        self._expand_noun_chunks = expand_noun_chunks

        self.last_sdp = self.last_sdp_root = None

    @property
    def channels(self):
        return len(self.vector_length)

    @property
    def vector_length(self):
        wv = self.wordvec_length
        ners = len(self.ner_tags)
        return [len(self.iob_tags), ners, len(self.pos_tags), len(self.dep_tags), len(self.wn_tags), wv, ners]

    @property
    def wordvec_length(self):
        return self.nlp.vocab.vectors_length

    @property
    def nbclasses(self):
        return len(self.tags)

    def __call__(self, crecords):
        for crecord in crecords:
            yield self.encode(crecord)

    def encode_data(self, s_span, o_span):
        """
        Encode data having entity spans (underlying doc is used implicitly by the means of dependency tree traversal)
        :param s_span: subject span
        :param o_span: object span
        :return: encoded data (tuple of arrays)
        """
        sdp = self._encode_sdp(s_span, o_span)

        _iob_tags = []
        _ner_tags = []
        _pos_tags = []
        _dep_tags = []
        _wn_hypernyms = []
        _vectors = []
        for t in sdp:
            _iob_tags.append(self.iob_tags.encode(t.ent_iob_))
            _ner_tags.append(self.ner_tags.encode(t.ent_type_))
            _pos_tags.append(self.pos_tags.encode(t.pos_))
            _wn_hypernyms.append(self.wn_tags.encode(get_hypernym_cls(t)))
            _vectors.append(t.vector)
            # Dependency tags by spacy
            dep_vars = t.dep_.split('||')
            if len(dep_vars) > 1:
                log.info('DBPediaEncoder: dep_tags: strange dep: token:"{}" dep:"{}"'.format(t.text, t.dep_))
            # dep = self.dep_tags.encode(dep_vars[0])
            # Use all dep types provided by dep parser...
            dep = sum(np.array(self.dep_tags.encode(dep_var)) for dep_var in dep_vars)
            _dep_tags.append(dep)

        s_type = self.ner_tags.encode(s_span.label_)
        o_type = self.ner_tags.encode(o_span.label_)
        s_type_out = np.array(self._encode_ent_position(s_span, s_type, np.zeros_like(s_type)))
        o_type_out = np.array(self._encode_ent_position(o_span, o_type, np.zeros_like(o_type)))

        data = _iob_tags, _ner_tags, _pos_tags, _dep_tags, _wn_hypernyms, _vectors, s_type_out + o_type_out
        return tuple(map(np.array, data))

    def _encode_sdp(self, s_span, o_span):
        # if entities are too close to each other, then add context around them to sdp
        ctx = 0
        if min(abs(o_span.start - s_span.end), abs(s_span.start - o_span.end)) < self._min_entities_dist:
            ctx = self._expand_context

        sdp, iroot = shortest_dep_path(s_span, o_span, include_spans=True, nb_context_tokens=ctx)
        sdp = expand_ents(sdp, self._expand_noun_chunks)
        self.last_sdp = sdp  # for the case of any need to look at that (as example, for testing)
        self.last_sdp_root = iroot

        return sdp

    def _encode_ent_position(self, ent_span, ent_indicator, zero_indicator):
        ent_indices = {token.i for token in ent_span}
        res = [ent_indicator if token.i in ent_indices else zero_indicator for token in self.last_sdp]
        return res

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
        s_span, o_span = crecord2spans(crecord, self.nlp, self.ntr)
        data = self.encode_data(s_span, o_span)
        cls = self.encode_class(crecord)
        return (*data, *cls)


class DBPediaEncoderEmbed(DBPediaEncoder):
    def encode_data(self, s_span, o_span):
        """
        Encode data having entity spans (underlying doc is used implicitly by the means of dependency tree traversal)
        :param s_span: subject span
        :param o_span: object span
        :return: encoded data (tuple of arrays)
        """
        sdp = self._encode_sdp(s_span, o_span)
        # todo: temporary to try returning full sentences
        # self.last_sdp = sdp = s_span.sent

        _iob_tags = []
        _ner_tags = []
        _pos_tags = []
        _dep_tags = []
        _wn_hypernyms = []
        _vectors = []
        for t in sdp:
            _iob_tags.append(self.iob_tags.encode_index(t.ent_iob_))
            _ner_tags.append(self.ner_tags.encode_index(t.ent_type_))
            _pos_tags.append(self.pos_tags.encode_index(t.pos_))
            _dep_tags.append(self.dep_tags.encode_index(t.dep_.split('||')[0]))
            _wn_hypernyms.append(self.wn_tags.encode_index(get_hypernym_cls(t)))
            _vectors.append(t.vector)

        s_type = self.ner_tags.encode(s_span.label_)
        o_type = self.ner_tags.encode(o_span.label_)
        s_type_out = np.array(self._encode_ent_position(s_span, s_type, np.zeros_like(s_type)))
        o_type_out = np.array(self._encode_ent_position(o_span, o_type, np.zeros_like(o_type)))

        data = _iob_tags, _ner_tags, _pos_tags, _dep_tags, _wn_hypernyms, _vectors, s_type_out + o_type_out
        return tuple(map(np.array, data))


class DBPediaEncoderBranched(DBPediaEncoderEmbed):
    @property
    def vector_length(self):
        _vl = super().vector_length
        return (*_vl, *_vl)

    def encode_data(self, s_span, o_span):
        data = super().encode_data(s_span, o_span)

        i = self.last_sdp_root
        if not (0 <= i < len(self.last_sdp)):
            i = 0
        left_parts = [d[:i+1] for d in data]
        right_parts = [d[i:] for d in data]
        return left_parts + right_parts

