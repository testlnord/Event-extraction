from collections import defaultdict

from spacy.matcher import Matcher
from spacy.tokens import Doc
import spacy.attrs as a

from experiments.extraction.extraction import Extraction, Part


class MetaExtractor:
    def __init__(self, matcher):
        self.matcher = matcher

    def match(self, extraction):
        # assert(isinstance(sent, Doc))
        self._e = extraction.dict
        return self.matcher(extraction.span)

    def intersect_acceptor(self, doc, ent_id, label, start, end, lm=0, rm=0):
        start1, end1 = self._e[label]['span']
        if (end1+rm > start) and (end > start1-lm):
            return (ent_id, label, start, end)

    def inclusion_acceptor(self, doc, ent_id, label, start, end, lm=0, rm=0):
        start1, end1 = self._e[label]['span']
        if (start1-lm <= start) and (end <= end1+rm):
            return (ent_id, label, start, end)

    def make_intersect_ar(self, left_margin=0, right_margin=0):
        return lambda a,b,c,d,e: self.intersect_acceptor(a, b, c, d, e, left_margin, right_margin)

    def make_inclusion_ar(self, left_margin=0, right_margin=0):
        return lambda a,b,c,d,e: self.inclusion_acceptor(a, b, c, d, e, left_margin, right_margin)


class Extractor(MetaExtractor):
    version = 1

    def __init__(self, nlp):
        matcher = Matcher(nlp.vocab)

        iob_pattern = [
            {a.LIKE_NUM: False, a.ENT_IOB: 3},
            {'OP': '*', a.ENT_IOB: 1},
            {'OP': '?', a.LIKE_NUM: True}
        ]
        entity_name = 'object'  # it is to associate matches with patterns
        matcher.add_entity(entity_name, acceptor=self.make_intersect_ar(2,2))
        matcher.add_pattern(entity_name, iob_pattern, label=Part.OBJ)

        entity_name = 'subject'
        matcher.add_entity(entity_name, acceptor=self.make_intersect_ar())
        matcher.add_pattern(entity_name, iob_pattern, label=Part.SUBJ)

        # conjugation_pattern = iob_pattern + [{a.POS: 'CONJ'}]

        # entity_name = 'version'
        # ver_pattern1 = [{a.LEMMA: 'version'}, {a.LIKE_NUM: True}]
        # matcher.add_entity(entity_name, acceptor=self.make_intersect_ar(1,1))
        # matcher.add_pattern(entity_name, ver_pattern1, label=Part.SUBJ)
        # matcher.add_pattern(entity_name, ver_pattern1, label=Part.OBJ)

        # entity_name = 'location'
        # entity_name = 'date'
        # matcher.add_entity(entity_name, acceptor=self.make_inclusion_ar(1,1))
        # matcher.add_pattern(entity_name, [{a.ENT_TYPE: 'DATE'}], label=Part.OBJ)

        self.entity_rules = ['subject', 'object']
        super().__init__(matcher)

    def process(self, extraction):
        relations = [(extraction.relation_span, extraction.relation)]  # just full relation
        # entities = []
        entities = defaultdict(list)
        attrs = {}
        unique = set()

        matches = self.match(extraction)
        # todo: it is possible here to associate extracted things with parts of extraction using 'label' variable
        for ent_id, label, start, end in matches:
            entity_name = self.matcher.vocab[ent_id].lower_  # ent_id by spacy logic is an index in Vocab
            end -= 1  # Matcher for some reason outputs 'end' higher (by 1) most of the times
            sp = (start, end)
            if sp not in unique:
                unique.add(sp)
                thing = extraction.span[start:end]
                if entity_name in self.entity_rules:
                    # entities.append((sp, thing))
                    entities[label].append((sp, thing))
                else:  # it is not an entity, not a relation, so, it is an attribute
                    attrs[entity_name] = (sp, thing)

        return dict(entities), relations, attrs


class ExtractionFilter:
    """
    Remove incoherent extractions
    """
    # todo: use language model?
    # todo: use confidence from generic KB?
    def process(self, entities, relations, attrs):
        return self._entities_ok(entities) and self._relation_ok(relations)

    def _entities_ok(self, entities):
        return all(len(part_ents) == 0 for part_ents in entities.values())

    def _relation_ok(self, relations):
        return any(any(self._verb_ok(token) for token in relation) for relation in relations)

    def _verb_ok(self, token):
        return token.pos_ == 'VERB'


# old
class NamedEntitiesFilter1:
    """
    In general, enhance coherency of extracted named entities.
    (e.g. add version numbers to software mentions)
    """
    # todo: heuristic for like "version xx OF yy" -- add yy to relation (or maybe "xx OF yy")
    # todo: process conjugated named entities -- partially done by self.process with sufficient mdist

    # todo: do self.process for subject with mdist = 1 or 2

    # todo: abstract self.process and just make an iterator over the nearest tokens is subtree and a pattern-matcher on top of it?

    def process(self, extraction):
        e = extraction
        e.subject_span = self.process_part(e.span, e.subject, e.subject_span, dist=1)
        e.object_max_span = self.process_part(e.span, e.object_max, e.object_max_span, dist=4)

        # todo: Process prepositions like 'of', 'for', etc.
        # process only prep-s that 1) have a link to something interesting (e.g. NE or date)
        return e

    def process_part(self, full_span, part_span, part_span_offsets, dist):
        b, e = part_span_offsets  # begin, end
        children = list(part_span.subtree)

        # todo: if span is Doc, then subtree_start = obj_ch[0].i -- much simpler
        # subtree_start = self._i_span(obj_ch[0], full_span)  # just need to shift all indices
        subtree_start = children[0].i - full_span.start  # just need to shift all indices
        b_subtree = b - subtree_start
        e_subtree = e - subtree_start
        lefts = children[b_subtree - dist:b_subtree]
        rights = children[e_subtree:e_subtree + dist]

        # Adding named enitites, that are not too far in the dependency subtree
        # todo: make it for noun_chunks (for better coherency?)
        # todo: process only if it is not contained in other parts of extraction? -- important stuff
        bnew, enew = b, e
        # for i, token in zip(range(len(lefts)-1, 0, -1), lefts):
        for i, token in enumerate(reversed(lefts), 1):
            # named entities
            if token.ent_iob_ == 'B' or token.ent_iob_ == 'I' \
                    or token.like_num:  # numbers: product or version numbers (or something else)
                bnew = b - i

        for i, token in enumerate(rights, 1):
            if token.ent_iob_ == 'B' or token.ent_iob_ == 'I' \
                    or token.like_num:
                enew = e + i

        return bnew, enew

