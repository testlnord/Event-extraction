import logging as log
import os
from math import floor
import pickle
from collections import defaultdict, namedtuple
from itertools import permutations, chain, zip_longest

from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor

from rdflib.namespace import RDF, RDFS
from pygtrie import StringTrie, CharTrie
from cytoolz import groupby, first, second
from fuzzywuzzy import fuzz


from experiments.ontology.sub_ont import NERTypeResolver
from experiments.ontology.sub_ont import get_label, get_fellow_disambiguations, get_fellow_redirects, dbo, gdbo


TrieEntry = namedtuple('TrieEntry', ['uri', 'sf', 'ent_type'])


class NERLinker:
    """
    Disambiguates named entities and stores new unknown if they satisfy type restrictions.
    """
    def __init__(self, ner_type_resolver, metric_threshold=0.8):
        self.ntr = ner_type_resolver

        # Init storage
        self._trie = CharTrie()
        self._metric_threshold = metric_threshold

    def update(self, uri_sf_pairs):
        """
        :param uri_sf_pairs: List[Tuple[URIRef, Option[str]]]: tolerable to None surface forms
        :return:
        """
        uri2sf = groupby(first, uri_sf_pairs)  # group by the same uri
        uris = list(uri2sf.keys())

        with Pool() as pool:
            def mmap(f, it): return list(map(f, it))  # todo: pool.map doesn't work: pickle issues with decorators
            ent_types = mmap(self.ntr.get_by_uri, uris)
            labels = mmap(get_label, uris)
            all_redirects = mmap(get_fellow_redirects, uris)  # lists of synonyms for each uri
            all_disambigs = mmap(get_fellow_disambiguations, uris)  # lists of omonyms for each uri

            for i, (ent_uri, ent_type, base_label, redirects, disambigs) in enumerate(zip(uris, ent_types, labels, all_redirects, all_disambigs), 1):
                if ent_type:
                    entries = {TrieEntry(ent_uri, base_label, ent_type)}  # basic entry
                    entries.update(TrieEntry(ent_uri, sf, ent_type) for sfs in uri2sf[ent_uri] for sf in sfs)  # entries from provided surface forms

                    redirects_labels = mmap(get_label, redirects)
                    entries.update(TrieEntry(ent_uri, sf, ent_type) for sf in redirects_labels)

                    disambigs_labels = mmap(get_label, disambigs)
                    disambigs_types = mmap(self.ntr.get_by_uri, disambigs)
                    entries.update(TrieEntry(duri, dsf, dtype) for duri, dsf, dtype in zip(disambigs, disambigs_labels, disambigs_types))

                    entries = filter(all, entries)  # all fields of entry should evaluate to True
                    sfgroups = groupby(lambda entry: entry.sf.lower(), entries)  # build 'index' of trie
                    _new = _upd = 0
                    for sfkey, group in sfgroups.items():
                        if not self._trie.has_key(sfkey):
                            self._trie[sfkey] = set(group)
                            _new += 1
                        else:
                            self._trie[sfkey].update(group)
                            _upd += 1
                    log.info('NERLinker: ent #{}: added {:3d}, updated {:3d} sfgroups; "{}"'.format(i, _new, _upd, str(ent_uri)))


    def _is_acronym(self, text, len_threshold=5):
        return len(text) < len_threshold and text.isupper()

    def _longest_prefix(self, text):
        l = len(text)
        left = max(1, floor(l * self._metric_threshold))
        for end in range(l, left-1, -1):
            if self._trie.has_subtrie(text[:end]):
                return text[:end]

    def get_uri(self, span):
        """
        Get URIRef of span or None if it is unknown entity.
        :param span: spacy.token.Span
        :return: URIRef or None
        """
        # if self._is_acronym(text): text = span.text  # not-acronyms should be searched by lower-case form

        text = span.text.lower()
        _trie = self._trie

        if span.label_ == 'PERSON':
            # If ner type is Person: try all permutations of tokens
            tokens = filter(bool, text.split(' '))
            lprefixes = [self._longest_prefix(' '.join(p))[0] for p in permutations(tokens)]
            lprefixes = filter(None, lprefixes)
            lprefix = max(lprefixes, key=len, default=None)
        else:
            lprefix, _ = self._longest_prefix(text)

        if lprefix is not None:
            candidate_sets = _trie.itervalues(prefix=lprefix)
            candidates = list(chain.from_iterable(candidate_sets))
            if len(candidates) > 0:
                return self._fuzzy_search(span, candidates)

    def _fuzzy_search(self, span, candidates, metric=fuzz.ratio):
        """

        :param span: spacy.token.Span
        :param candidates: List[TrieEntry]
        :return: best-matching URIRef or None
        """
        typed = groupby(lambda entry: entry.type == span.label_, candidates)
        # Search with the same type first
        for is_same_type in [True, False]:
            _candidates = typed.get(is_same_type)
            if _candidates:
                measured = [(metric(entry.sf, span.text), entry) for entry in candidates]
                m, entry = max(measured, key=second)
                if m > self._metric_threshold * 100:
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
            self._trie = pickle.load(f)


def from_graph(graph):
    yield from zip_longest(set(graph.subjects()), [])  # zip with [None] * len(...)


def from_crecords(crecords):
    for record in crecords:
        for ent in record.ents:
            uri = ent.uri
            surface_form = ent.text
            if uri and surface_form:
                surface_form = ent.text.strip(' \n\t,-:;').replace('"', '')
                surface_form = max(surface_form.split('\n'), key=len)  # stripping extra lines
                if len(surface_form) > 0:
                    yield ent.uri, surface_form


def test_pools():
    from multiprocessing import Pool
    from time import process_time, perf_counter

    from experiments.ontology.sub_ont import gf

    ntr = NERTypeResolver()
    uris = list(gf.subjects(dbo.product))
    print(len(uris), uris)

    # workers = [64, 128, 256, 512]
    # for wk in workers:
    #     tstart = perf_counter()
    #     with ThreadPoolExecutor(max_workers=wk) as pool:
    #         ent_types = pool.map(ntr.get_by_uri, uris)
    #     tend = perf_counter()
    #     print('elapsed thread-pool({}): {}'.format(wk, tend - tstart))

    workers = [8, 16]

    for wk in workers:
        tstart = perf_counter()
        with ProcessPoolExecutor(max_workers=wk) as pool:
            ent_types = pool.map(ntr.get_by_uri, uris)
        tend = perf_counter()
        print('elapsed pool({}): {}'.format(wk, tend - tstart))


def build_linker():
    from experiments.ontology.sub_ont import gf, gfall, dbr
    from experiments.data_utils import unpickle

    base_dir = '/home/user/'
    project_dir = os.path.join(base_dir, 'projects', 'Event-extraction')
    model_dir = os.path.join(project_dir, 'experiments', 'models')
    assert os.path.isdir(model_dir)

    dataset_dir = '/home/user/datasets/dbpedia/ner/'
    dataset_file = 'crecords.v2.pck'
    crecords = list(unpickle(dataset_dir + dataset_file))
    data1 = set(from_crecords(crecords))
    # for i, (ent_uri, sf) in enumerate(data1, 1):
    #     print(i, str(ent_uri).ljust(80), str(sf))
    # exit()

    graph = gfall
    data2 = list(from_graph(graph))

    ntr = NERTypeResolver()
    linker = NERLinker(ner_type_resolver=ntr)
    for data in [data1, data2]:
        log.info('linker: updating data ({} records)'.format(len(data)))
        linker.update(data)
        linker.save(model_dir)

    test_linker(linker)
    linker.load(model_dir)
    test_linker(linker)


def try_trie():
    from experiments.ontology.sub_ont import gf, gfall, dbr

    base_dir = '/home/user/'
    project_dir = os.path.join(base_dir, 'projects', 'Event-extraction')
    model_dir = os.path.join(project_dir, 'experiments', 'models')

    ntr = NERTypeResolver()
    linker = NERLinker(ner_type_resolver=ntr)

    # subjs = list(gfall.subjects(dbr.Microsoft))
    subjs = [dbr.Alien, dbr.Microsoft, dbr.JetBrains, dbr.Google]
    data = list(zip_longest(subjs, []))
    # data = list(from_graph(gfall))

    print('total entities: {}'.format(len(data)))
    linker.update(data)
    print('total entities: {}'.format(len(data)))
    test_linker(linker)
    linker.save(model_dir)


def test_linker(linker):
    trie = linker._trie
    entry_sets = list(trie.itervalues())
    all_entries = set(chain.from_iterable(entry_sets))
    total_entries = sum(map(len, entry_sets))
    total_unique_entries = len(all_entries)
    print('TOTAL KEYS: {}; TOTAL ENTRIES: {} (UNIQUE: {})'.format(len(trie), total_entries, total_unique_entries))
    assert all(map(all, all_entries)), 'found bad entries: {}'.format(len([e for e in all_entries if not all(e)]))
    assert total_unique_entries == total_entries


if __name__ == "__main__":
    from experiments.ontology.data_structs import ContextRecord, EntityRecord  # for unpickle()
    log.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=log.INFO)

    # test_pools()
    # try_trie()

    build_linker()
