from intervaltree import Interval
from rdflib import URIRef


class RelRecord:
    def __init__(self, relation, s0: int, s1: int, o0: int, o1: int,
                 ctext, cstart=0, cend=None, source_id=None):
        self.relation = relation
        self.s0 = self.s_start = s0
        self.s1 = self.s_end = s1
        self.o0 = self.o_start = o0
        self.o1 = self.o_end = o1
        self.context = ctext
        self.cstart = cstart
        self.cend = cend if cend is not None else cstart + len(ctext)
        self.source_id = source_id if source_id is not None else hash(ctext)

    def cut_context(self, begin, end):
        self.context = self.context[begin:end]
        self.cend = self.cstart + end
        self.cstart = self.cstart + begin
        self.s0 = self.s_start = max(self.cstart, self.s0)
        self.s1 = self.s_end = min(self.s1, self.cend)
        self.o0 = self.o_start = max(self.cstart, self.o0)
        self.o1 = self.o_end = min(self.o1, self.cend)
        assert self.valid_offsets

    @property
    def direction(self):
        """
        Is direction of relation the same as order of (s, o) in the context.
        :return: None, True or False
        """
        return None if not bool(self.relation) else (self.s_end <= self.o_start)

    @property
    def valid_offsets(self):
        return ((self.cstart <= self.s_start < self.s_end <= self.cend) and
                (self.cstart <= self.o_start < self.o_end <= self.cend) and
                disjoint(self.s_span, self.o_span))

    @property
    def s_startr(self): return self.s_start - self.cstart  # offsets in dataset are relative to the whole document, not to the sentence

    @property
    def s_endr(self): return self.s_end - self.cstart

    @property
    def o_startr(self): return self.o_start - self.cstart

    @property
    def o_endr(self): return self.o_end - self.cstart

    @property
    def s_span(self): return (self.s_start, self.s_end)

    @property
    def o_span(self): return (self.o_start, self.o_end)

    @property
    def s_spanr(self): return (self.s_startr, self.s_endr)

    @property
    def o_spanr(self): return (self.o_startr, self.o_endr)

    @property
    def subject_text(self): return self.context[self.s_startr: self.s_endr]

    @property
    def object_text(self): return self.context[self.o_startr: self.o_endr]

    @property
    def triple(self): return (self.subject_text, self.relation, self.object_text)

    def __hash__(self): return hash(self.id)

    def __eq__(self, other): return self.id == other.id

    @property
    def id(self):
        return (self.source_id, (self.cstart, self.cend),
                self.s_span, self.o_span, self.triple)

    def __str__(self):
        return '\n'.join((' '.join('<{}>'.format(x) for x in self.triple), self.context.strip()))


class RelationRecord(RelRecord):
    def __init__(self, s: URIRef, r: URIRef, o: URIRef,
                 s0: int, s1: int, o0: int, o1: int,
                 ctext, cstart=0, cend=None, source_id=None):
        self.subject = s
        self.relation = r
        self.object = o
        super().__init__(r, s0, s1, o0, o1, ctext, cstart, cend, source_id)

    @property
    def triple(self): return (self.subject, self.relation, self.object)


class EntityRecord:
    def __init__(self, crecord, start, end, uri):
        """

        :param uri: original uri
        :param start: char offset of the start in crecord.context (including start)
        :param end: char offset of the end in crecord.context (not including end)
        :param crecord: source of entity: ContextRecord
        """
        self.crecord = crecord
        self.start = start
        self.end = end
        self.uri = uri

    @property
    def text(self):
        return self.crecord.context[slice(*self.span)]

    @property
    def span(self):
        return self.start, self.end

    @property
    def spang(self):
        s = self.crecord.start
        return s+self.start, s+self.end

    def cut_context(self, begin, end):
        self.start = max(0, self.start - begin)
        self.end = min(self.end, end)

    def json(self):
        return self.span, self.uri

    def __str__(self):
        return '[{}:{}] {}'.format(self.start, self.end, self.text.strip())


class ContextRecord:
    @classmethod
    def from_span(cls, span, artid, ents=None):
        return cls(span.text, span.start_char, span.end_char, artid, ents)

    def __init__(self, ctext, cstart, cend, artid, ents=None):
        self.context = ctext
        self.start = cstart
        self.end = cend
        self.article_id = artid
        self.ents = [] if ents is None else ents

    def cut_context(self, begin, end):
        self.context = self.context[begin:end]
        self.end = self.start + end
        self.start = self.start + begin
        for e in self.ents:
            e.cut_context(begin, end)

    @property
    def span(self):
        return self.start, self.end

    def json(self):
        return (self.article_id, self.span, self.context, [e.json for e in self.ents])

    def __str__(self):
        return self.context.strip() + '(' + '; '.join(str(e) for e in self.ents) + ')'


def disjoint(span1, span2):
    return not Interval(*span1).overlaps(*span2)