from spacy.tokens import Span, Token
from enum import Enum, IntEnum


class Part(IntEnum):
    # SUBJ = 'subj'
    # REL = 'rel'
    # OBJ_MAX = 'obj_max'
    # OBJ_MIN = 'obj_min'
    # OBJ = 'obj'
    SUBJ = 0
    REL = 1
    OBJ = 2
    OBJ_MAX = 3
    OBJ_MIN = 4
    DATE = 10


class Extraction:
    def __init__(self, span: Span, subject_span, relation_span,
                 object_min_span, object_max_span):
        self.span = span
        self.subject_span = subject_span
        self.relation_span = relation_span
        self.object_max_span = object_max_span
        self.object_min_span = object_min_span
        self.object_span = object_max_span

    @property
    def dict(self):
        return {
            Part.SUBJ: {'span': self.subject_span, 'it': self.subject},
            Part.REL: {'span': self.relation_span, 'it': self.relation},
            Part.OBJ_MAX: {'span': self.object_max_span, 'it': self.object_max},
            Part.OBJ_MIN: {'span': self.object_min_span, 'it': self.object_min},
            Part.OBJ: {'span': self.object_span, 'it': self.object},
        }

    @property
    def subject(self):
        return self._substring(self.subject_span)

    @property
    def relation(self):
        return self._substring(self.relation_span)

    @property
    def object_max(self):
        return self._substring(self.object_max_span)

    @property
    def object_min(self):
        return self._substring(self.object_min_span)

    @property
    def object(self):
        return self._substring(self.object_span)

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
    def __init__(self, span: Span, subject_indices, relation_indices, object_indices):
        self.span = span
        self._subject = subject_indices
        self._relation = relation_indices
        self._object = object_indices

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
