import logging as log
from collections import defaultdict, namedtuple
from itertools import permutations

from rdflib.namespace import RDF, RDFS
from pygtrie import StringTrie, CharTrie
from cytoolz import groupby, first, second
from fuzzywuzzy import fuzz


from experiments.ontology.sub_ont import get_label, dbo, gdbo


TrieEntry = namedtuple('TrieEntry', ['uri', 'sf', 'ent_type'])


class NERLinker:
    """
    Disambiguates named entities and stores unknown if they satisfy type restrictions.
    """
    # todo: store built model
    @classmethod
    def from_storage(cls, path):
        pass

    def __init__(self, allowed_classes, ner_type_resolver, outer_graph, rdf_namespace=dbo):
        # self.outer_graph = rdf_dataset.get_context(str(rdf_namespace))
        self.outer_graph = outer_graph
        self.ns = rdf_namespace
        self.ntr = ner_type_resolver

        # Init storages
        self._trie = CharTrie()

        _classes = set(allowed_classes)
        for uri, _, type_uri in gdbo.triples((None, RDFS.type, None)):  # there should not be duplicates
            # todo: synchronously add to stores each asynchronously generated list of synonyms  (Pool.async_map or like that)

            ent_type = ner_type_resolver.get_by_type_uri(type_uri)  # todo: ensure no extra work or side effect is done
            if ent_type in _classes:  # case when ent_type is None is handled
                base_label = get_label(uri)
                synonyms = [base_label]
                # -disambigs?
                # -redirects
                # -acronyms
                # todo: ensure synonyms have no None (or otherwise incorrect) entries
                # Storing in trie upper-case forms for acronyms and lower-case forms for all other words
                keys = [syn.lower() if not self._is_acronym(syn) else syn for syn in synonyms]
                self._trie.update({key: TrieEntry(uri, syn, ent_type) for key, syn in zip(keys, synonyms)})

    def _is_acronym(self, text, len_threshold=5):
        return len(text) < len_threshold and text.isupper()

    def get_uri(self, span):
        """
        Get URIRef of span or None if it is unknown entity.
        :param span: spacy.token.Span
        :return: URIRef or None
        """
        text = span.text
        if not self._is_acronym(text): text = text.lower()  # not-acronyms should be searched by lowe-case form
        _trie = self._trie

        if span.label_ == 'PERSON':
            # If ner type is Person: try all permutations of tokens
            tokens = filter(bool, text.split(' '))
            lprefixes = [_trie.longest_prefix(' '.join(p))[0] for p in permutations(tokens)]
            lprefixes = list(filter(None, lprefixes))
            lprefix = max(lprefixes, key=len, default=None)
        else:
            lprefix, _ = _trie.longest_prefix(text)

        if lprefix is not None:
            candidates = list(_trie.itervalues(prefix=lprefix))
            if len(candidates) > 0:
                return self._fuzzy_search(span, candidates)

    def _fuzzy_search(self, span, candidates, metric=fuzz.ratio, metric_threshold=82):
        """

        :param span: spacy.token.Span
        :param candidates: List[TrieEntry]
        :return: best-matching URIRef or None
        """
        typed = groupby(lambda entry: entry.type == span.label_, candidates)
        text = span.text
        # Search using corresponding type first
        for _candidates in (typed[True], typed[False]):
            if len(_candidates) > 0:
                measured = [(metric(entry.text, text), entry) for entry in candidates]
                m, entry = max(measured, key=second)
                if m > metric_threshold:
                    return entry.uri

    def add_ent(self, new_ent):
        pass
