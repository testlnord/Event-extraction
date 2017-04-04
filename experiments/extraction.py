from spacy.tokens import Span, Token


class Extraction:
    def __init__(self, span: Span, subject_offsets, relation_offsets, object_offsets,
                 source=None):
        assert(isinstance(span, Span))

        self.span = span
        self.subject = self._substring(subject_offsets)
        self.relation = self._substring(relation_offsets)
        self.object = self._substring(object_offsets)

    def _substring(self, offset_pair):
        beg, end = offset_pair
        return self.span[beg:end]

    def __str__(self):
        return '; '.join(map(lambda span: span.text,
                            [self.subject, self.relation, self.object]))

class Extraction3:
    def __init__(self, span: Span, subject_offsets, relation_offsets,
                 object_min_span, object_max_span,
                 source=None):
        assert(isinstance(span, Span))

        self.span = span
        self.subject = self._substring(subject_offsets)
        self.relation = self._substring(relation_offsets)
        self.object_min = self._substring(object_min_span)
        self.object_max = self._substring(object_max_span)
        self.object = self.object_max

    def _substring(self, offset_pair):
        beg, end = offset_pair
        return self.span[beg:end]

    def __str__(self):
        return '; '.join(map(lambda span: span.text,
                             [self.subject, self.relation, self.object]))
    # todo: remove then
    @property
    def str2(self):
        return '; '.join(map(lambda span: span.text,
                             [self.subject, self.relation, self.object_min]))

class Extraction2:
    def __init__(self, span: Span, subject_indices, relation_indices, object_indices,
                 source=None):
        assert(isinstance(span, Span))

        self.span = span
        self._subject = subject_indices
        self._relation = relation_indices
        self._object = object_indices

        # todo: how to represent this?
        self.source = None
        self.time = None
        self.location = None

    @property
    def subject(self):
        return [self.span[i] for i in self._subject]

    @property
    def object(self):
        return [self.span[i] for i in self._object]

    @property
    def relation(self):
        return [self.span[i] for i in self._relation]

    def __str__(self):
        return '; '.join(map(lambda span: span.text,
                             [self.subject, self.relation, self.object]))
