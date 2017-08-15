import logging as log
import os
import pickle
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
    Disambiguates named entities and stores new unknown if they satisfy type restrictions.
    """
    def __init__(self, ner_type_resolver, outer_graph, rdf_namespace=dbo):
        self.ntr = ner_type_resolver
        # self.outer_graph = rdf_dataset.get_context(str(rdf_namespace))
        self.outer_graph = outer_graph  # todo: is it needed?
        self.ns = rdf_namespace

        # Init storages
        self._trie = CharTrie()

    def update(self, uri_sf_pairs):
        uri2sf = groupby(uri_sf_pairs, first)

        # todo: parallelize (try thread-based)
        uris = list(uri2sf.keys())
        ent_types = map(self.ntr.get_by_uri, uris)

        for ent_type, ent_uri in zip(ent_types, uris):
            if ent_type:
                base_label = get_label(ent_uri)
                surface_forms = [base_label]
                surface_forms.extend(uri2sf[ent_uri])

                # -disambigs?
                # -redirects
                # -acronyms
                # todo: ensure surface_forms have no None (or otherwise incorrect) entries
                # Storing in trie upper-case forms for acronyms and lower-case forms for all other words
                sfkeys = [sf if self._is_acronym(sf) else sf.lower() for sf in surface_forms]
                self._trie.update({sfkey: TrieEntry(ent_uri, sf, ent_type) for sfkey, sf in zip(sfkeys, surface_forms)})

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

    # todo:
    def add_ent(self, new_ent):
        pass

    def save(self, model_dir):
        model_name = type(self).__name__.lower() + '.pck'
        with open(os.path.join(model_dir, model_name), 'wb') as f:
            pickle.dump(self._trie, f)

    def load(self, model_dir):
        model_name = type(self).__name__.lower() + '.pck'
        with open(os.path.join(model_dir, model_name), 'rb') as f:
            self._trie = pickle.load(self._trie, f)


def from_graph(graph):
    yield from graph.subjects()


def from_crecords(crecords):
    # any need to filter them?
    for record in crecords:
        text = record.context
        for a, b, ent_uri in record.ents:
            surface_form = text[a:b]
            if surface_form:
                yield ent_uri, surface_form


if __name__ == "__main__":
    log.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=log.INFO)

    from experiments.ontology.sub_ont import NERTypeResolver, gdb

    base_dir = '/home/user/'
    project_dir = os.path.join(base_dir, 'Event_extraction', 'experiments')
    model_dir = os.path.join(project_dir, 'models')

    ntr = NERTypeResolver()
    linker = NERLinker(ner_type_resolver=ntr, outer_graph=gdb)
    # linker.load(model_dir)
