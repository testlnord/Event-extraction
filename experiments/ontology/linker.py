import logging as log
import os
from math import floor
import pickle
from collections import defaultdict, namedtuple
from itertools import permutations, chain, zip_longest

from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor

import spacy
from rdflib import URIRef
from rdflib.namespace import RDF, RDFS
from pygtrie import CharTrie
from cytoolz import groupby, first, second
from fuzzywuzzy import fuzz
import networkx as nx

from experiments.ontology.symbols import ENT_CLASSES
from experiments.ontology.sub_ont import get_label, get_fellow_disambiguations, get_fellow_redirects, dbo, gdbo, \
    get_type, get_superclass, raw, raw_d, gdb


class NERTypeResolver:
    from experiments.ontology.symbols import ENT_MAPPING
    # Map uris of final classes to its' names
    final_classes_names = {URIRef(dbo[s]): (ent_type if ent_type is not None else s) for s, ent_type in ENT_MAPPING.items()}

    def __init__(self, raw_classes=tuple()):
        assert(all(isinstance(x, URIRef) for x in raw_classes))
        self.superclasses_map = self.get_superclasses_map(raw_classes)

    def get_by_uri(self, uri, default_type=None):
        return self.get_by_type_uri(get_type(uri), default_type)

    def get_by_type_uri(self, type_uri, default_type=None):
        final_type_uri = self.superclasses_map.get(type_uri, self.get_final_class(type_uri))  # try to get the type from buffer
        if final_type_uri is not None:
            self.superclasses_map[type_uri] = final_type_uri  # add type to buffer (or do nothing useful if it is already there)
            return self.final_classes_names[final_type_uri]
        return default_type

    def get_final_class(self, cls):
        c = cls
        while not(c in self.final_classes_names or c is None):
            c = get_superclass(c)
            if c == cls:
                return None
            cls = c
        return c

    def get_superclasses_map(self, classes):
        superclasses = dict()
        for cls in classes:
            fc = self.get_final_class(cls)
            if fc is not None:
                superclasses[cls] = fc
        return superclasses


TrieEntry = namedtuple('TrieEntry', ['uri', 'sf', 'ent_type'])


class NERLinker:
    """
    Disambiguates named entities and stores new unknown if they satisfy type restrictions.
    """
    def __init__(self, outer_graph=gdb, ner_type_resolver=NERTypeResolver(),
                 metric_threshold=0.8, strict_type_match=True):
        self.ntr = ner_type_resolver

        # Init storage
        self._trie = CharTrie()
        self._metric_threshold = metric_threshold
        self._strict_type_match = strict_type_match
        self._allowed_types = ENT_CLASSES

        self.outer_graph = outer_graph

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

    def get_path_graph1(self, uris):
        graph = nx.DiGraph()
        for s, o in permutations(uris, 2):
            for s_uri, r_uri, o_uri in self.outer_graph.triples((s, None, o)):
                graph.add_edge(s_uri, o_uri)
                # graph.add_edge(s_uri, o_uri, {'relation': r_uri})  # add to graph
        return graph

    def get_path_graph(self, uris, depth=2):
        ont_graph = self.outer_graph
        edges = list()
        new_nodes = list(uris)
        for i in range(depth):
            for uri in new_nodes:
                objs = list(ont_graph.objects(subject=uri, predicate=None))
                subjs = list(ont_graph.subjects(object=uri, predicate=None))
                edges.extend((uri, obj) for obj in objs)
                edges.extend((subj, uri) for subj in subjs)
                new_nodes = objs + subjs

        graph = nx.DiGraph()
        graph.add_edges_from(edges)
        # todo: is it needed? maybe return the full graph for HITS algo...
        subgraph = nx.transitive_closure(graph).subgraph(nbunch=uris)

        return subgraph

    def link(self, ents):
        # todo: return uris or entries?
        answers = {ent: None for ent in ents}
        # Get candidate sets for all ents
        candidates = {cand.uri: ent for ent in ents for cand in self.get_candidates(ent)}
        # Build subgraph for these candidates
        graph = self.get_path_graph(candidates, depth=3)
        # Apply HITS or PageRank algorithm
        hubs, authorities = nx.hits(graph, max_iter=20)
        # Sort according to authority value
        authorities = sorted(authorities.items(), key=second)

        for uri, auth_value in authorities:
            ent = candidates.get(uri)
            if ent and not answers[ent]:
                answers[ent] = uri
        return answers

    def __call__(self, doc):
        """
        Resolve uris of all entities, remember resolved uris;
        correct entity types in doc for resolved entities;
        ? remember new entities -- candidates for adding to ontology.
        :param doc: spacy.Doc
        :return: modified @doc
        """

        from spacy.tokens import Span
        string_store = doc.vocab.strings

        new_entities = []
        for ent in doc.ents:

            # Resolve all entities:
            #   get types, (and flag whether type the same?), get confidence score
            # !Is there any possibility to get type-guess or confidence to unknown entities?
            #   somehow remember new entities as candidates, dump them somewhere
            # Remember resolved somehow, link with Doc somehow
            # Get resolved with different from spacy.ents types
            # Merge them giving appropriate priority (use confidence, etc.)
            # Set doc.ents

            # todo: get types also
            trie_entry = self.get_candidates(ent)
            ent_type = trie_entry.ent_type

            if trie_entry:
                # todo: cache resolved?

                # Change type annotations: priority to our type
                label_id = string_store[ent_type]
                new_entities.append(Span(doc=doc, start=ent.start, end=ent.end, label=ent_type))

                # todo: we may check if spacy types and our guessed types are the same
                #   but is it needed?
                if not self._strict_type_match:
                    guess_type = self.ntr.get_by_type_uri(ent_type)
                    if guess_type and guess_type == ent.label_:
                        pass
            else:
                # todo: remember unknown entity?

                new_entities.append(ent)

        doc.ents = tuple(new_entities)
        return doc

    def get_candidates(self, span):
        """

        :param span: spacy.token.Span
        :return: List[TrieEntry]
        """
        _trie = self._trie
        text = span.text.lower()
        candidates_filtered = []
        if span.label_ in self._allowed_types:
            # Determine how it's better to search
            if span.label_ == 'PERSON':
                # If ner type is Person: try all permutations of tokens
                tokens = filter(bool, text.split(' '))
                lprefixes = [self._longest_prefix(' '.join(p)) for p in permutations(tokens)]
                lprefixes = filter(bool, lprefixes)
                lprefix = max(lprefixes, key=len, default=None)
            else:
                lprefix = self._longest_prefix(text)

            if lprefix is not None:
                # log.info('span: "{}"; found prefix: "{}"'.format(span, lprefix))
                candidate_sets = _trie.itervalues(prefix=lprefix)
                candidates = list(chain.from_iterable(candidate_sets))

                # todo: merge both typed and non-typed somehow, i.e. give some precedence, weight?
                typed = groupby(lambda entry: entry.ent_type == span.label_, candidates)
                search_in = [True]
                if not self._strict_type_match:
                    search_in.append(False)

                # Search with the same type first
                for is_same_type in search_in:
                    typed_candidates = typed.get(is_same_type)
                    if typed_candidates:
                        candidates_filtered.extend(self._fuzzy_filter(span.text, typed_candidates))
        return candidates_filtered  # in the case of not-found just the empty list

    def _fuzzy_filter(self, text, candidates, metric=fuzz.ratio):
        """
        :param text: str
        :param candidates: List[TrieEntry]
        :param metric: (str, str) -> Numeric
        :return: List[TrieEntry]
        """
        # similar = groupby(lambda entry: metric(entry.sf, text), candidates)  # group by val of metric
        # Calculate a metric
        measured = [(metric(entry.sf, text), entry) for entry in candidates]
        # Group by the same uri
        similar = groupby(lambda entry: entry[1].uri, measured)  # uri: (m, entry)
        # In each group of same matches leave only the one with the highest match-metric
        similar = [max(sames, key=first) for sames in similar.values()]
        # Sort by the metric
        best_matches = sorted(similar, key=first, reverse=True)
        # Filter bad matches
        best_matches = [entry for m, entry in best_matches if m >= self._metric_threshold * 100]

        # todo: do something with that, e.g. weight matches by that second metric and sort?
        # Some more checks on the best matches if there're several matches
        # if len(best_matches) > 1:
        #     best_match = max(best_matches, key=lambda entry: metric(raw_d(raw(entry.uri)), span.text))
        #     return [best_match]
        return best_matches

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


def build_linker(linker, model_dir):
    from experiments.ontology.sub_ont import gf, gfall, dbr
    from experiments.data_utils import unpickle
    dataset_dir = '/home/user/datasets/dbpedia/ner/'
    dataset_file = 'crecords.v2.pck'
    crecords = list(unpickle(dataset_dir + dataset_file))
    data1 = set(from_crecords(crecords))
    # for i, (ent_uri, sf) in enumerate(data1, 1):
    #     print(i, str(ent_uri).ljust(80), str(sf))
    # exit()

    graph = gfall
    data2 = list(from_graph(graph))

    for data in [data1, data2]:
        log.info('linker: updating data ({} records)'.format(len(data)))
        linker.update(data)
        linker.save(model_dir)
    test_linker_trie(linker)
    # linker.load(model_dir)
    # test_linker_trie(linker)


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
    test_linker_trie(linker)
    linker.save(model_dir)


def test_linker_trie(linker):
    trie = linker._trie
    entry_sets = list(trie.itervalues())
    all_entries = set(chain.from_iterable(entry_sets))
    total_entries = sum(map(len, entry_sets))
    total_unique_entries = len(all_entries)
    print('TOTAL KEYS: {}; TOTAL ENTRIES: {} (UNIQUE: {})'.format(len(trie), total_entries, total_unique_entries))
    assert all(map(all, all_entries)), 'found bad entries: {}'.format(len([e for e in all_entries if not all(e)]))
    assert total_unique_entries == total_entries


def test_linker(linker):
    from experiments.ontology.sub_ont import get_article, dbr

    # from experiments.ontology.data import nlp
    model_dir = '/home/user/projects/Event-extraction/experiments/ontology'
    # model_name = 'models.v5.4.i{}.epoch{}'.format(1, 8)
    model_name = 'models.v5.4.i{}.epoch{}'.format(5, 2)
    model_path = os.path.join(model_dir, model_name)
    nlp = spacy.load('en', path=model_path)

    titles = [dbr.JetBrains, dbr.Microsoft_Windows]
    for i, title in enumerate(titles):
        print('\n')
        print(i, str(title))
        article = get_article(title)
        text = nlp(article['text'])
        prev_sent = ''
        for ent in text.ents:
            sent_text = ent.sent.text.strip()
            if sent_text != prev_sent:
                print()
                print(sent_text)
                prev_sent = sent_text

            uris = linker.get_candidates(ent)
            uris_str = uris and '; '.join(map(raw, uris)) or uris
            print('({}:{}) ({}) <{}> is [{}]'.format(ent.start, ent.end, ent.label_, ent.text, uris_str))
            # print('({}:{}) ({}) <{}> is <{}>'.format(ent.start, ent.end, ent.label_, ent.text, str(uri)))


if __name__ == "__main__":
    from experiments.ontology.data_structs import ContextRecord, EntityRecord  # for unpickle()
    # from experiments.ontology.config import config
    from experiments.ontology.symbols import ENT_CLASSES

    log.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=log.INFO)

    # print(list(sorted(set(NERTypeResolver.final_classes_names.values()))))
    # print(list(sorted(set(ENT_CLASSES))))
    # exit()

    base_dir = '/home/user/'
    project_dir = os.path.join(base_dir, 'projects', 'Event-extraction')
    model_dir = os.path.join(project_dir, 'experiments', 'models')
    assert os.path.isdir(model_dir)

    ntr = NERTypeResolver()
    linker = NERLinker(ner_type_resolver=ntr, metric_threshold=0.8, strict_type_match=False)

    # build_linker(linker, model_dir)

    linker.load(model_dir)
    test_linker(linker)

    # test_pools()
    # try_trie()
