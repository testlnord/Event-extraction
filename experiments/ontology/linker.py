import logging as log
from math import floor
import pickle
from collections import defaultdict, namedtuple
from itertools import permutations, chain, zip_longest

from multiprocessing import Pool

import spacy
from rdflib import URIRef, Literal
from rdflib.namespace import RDF, RDFS
from pygtrie import CharTrie
from cytoolz import groupby, first, second
from fuzzywuzzy import fuzz
import networkx as nx

from experiments.ontology.config import config, load_nlp
from experiments.ontology.symbols import ENT_CLASSES, ENT_MAPPING
from experiments.ontology.sub_ont import *


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


# todo: remember about consistency between ENT_CLASSES, that nlp classes and Trie stored classes


class NERLinker:
    """
    Disambiguates named entities and stores new unknown if they satisfy type restrictions.
    """
    def __init__(self, outer_graph=gall, ner_type_resolver=NERTypeResolver(),
                 metric_threshold=0.8, strict_type_match=True):
        self.ntr = ner_type_resolver

        # Init storage
        self._trie = CharTrie()
        self._metric_threshold = metric_threshold
        self._strict_type_match = strict_type_match
        self._allowed_types = ENT_CLASSES

        self.predicate_namespace = dbo  # todo: move to constructor args
        self.outer_graph = outer_graph
        self.cache = dict()

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

    def _resolve_edges(self, target_nodes, source_nodes):
        ont_graph = self.outer_graph
        ns = self.predicate_namespace
        new_edges = set()
        targets = set(target_nodes)

        for s in source_nodes:
            for rel, obj in ont_graph.predicate_objects(subject=s):
                if obj in targets and rel.startswith(ns) and obj != s:
                    new_edges.add((s, obj))
        return new_edges

    def _resolve_nodes(self, uri):
        ont_graph = self.outer_graph
        ns = self.predicate_namespace

        objs = {obj for rel, obj in ont_graph.predicate_objects(subject=uri)
                if rel.startswith(ns) and not isinstance(obj, Literal)}
        subjs = {subj for subj, rel in ont_graph.subject_predicates(object=uri)
                 if rel.startswith(ns)}

        new_edges = {(uri, obj) for obj in objs}.union({(subj, uri) for subj in subjs})
        new_nodes = objs.union(subjs)
        return new_nodes, new_edges

    def get_path_graph(self, uris, depth=2):
        """
        Based on the paper: https://arxiv.org/pdf/1707.05288.pdf
        :param uris: uris to build graph for
        :param depth: depth of the paths to search
        :return:
        """
        edges = set()
        nodes = set(uris)
        log.info('linker: started building subgraph on {} nodes with depth {}'.format(len(nodes), depth))

        mmap = map  # todo: make parallel queries
        for i in range(depth - 1):
            new_nodes = set()
            for uri_nodes, uri_edges in mmap(self._resolve_nodes, nodes):
                new_nodes.update(uri_nodes)
                edges.update(uri_edges)
            nodes = new_nodes
            log.info('linker: finished iter {}/{} with {} new nodes, {} edges'.format(i+1, depth, len(new_nodes), len(edges)))

        # Last step can be done easier
        edges.update(self._resolve_edges(uris, nodes))
        log.info('linker: finished building subgraph: {} edges'.format(len(edges)))

        graph = nx.DiGraph()
        graph.add_nodes_from(uris)  # need only original entities
        graph.add_edges_from(edges)

        subgraph = nx.transitive_closure(graph).subgraph(nbunch=uris)
        log.info('linker: ended extracting subgraph: {} edges'.format(len(subgraph.edges())))

        return subgraph

    def link(self, ents, depth=2):
        """

        :param ents:
        :return: Dict[spacy.token.Span, rdflib.URIRef]
        """
        answers = {ent: None for ent in ents if ent.label_ in self._allowed_types}
        # Get candidate sets for all ents
        all_candidates = [(cand.uri, ent) for ent in ents for cand in self.get_candidates(ent)]
        # Each candidate can resolve multiple entities
        candidates = defaultdict(list)
        for cand_uri, ent in all_candidates:
            candidates[cand_uri].append(ent)

        # Build subgraph for these candidates
        graph = self.get_path_graph(candidates, depth=depth)
        # Apply HITS or PageRank algorithm
        hubs, authorities = nx.hits(graph, max_iter=20)
        # Sort according to authority value
        authorities = sorted(authorities.items(), key=second, reverse=True)

        # todo: what to do with equally probable authorities? or with 'zero' authorities?
        #   maybe somehow preserve initial sort by get_candidates()? or returned weights (if any)
        for uri, auth_value in authorities:
            ents = candidates.get(uri, list())
            for ent in ents:
                if not answers[ent]:
                    answers[ent] = uri
        return answers

    def __call__(self, doc):
        answer_lists = {ent: self.get_candidates(ent) for ent in doc.ents}
        self.cache.update({ent: [str(entry.uri) for entry in answers] for ent, answers in answer_lists.items()})

        return doc

    def get(self, span, default=None):
        return self.cache.get(span, None) or default  # cache can return some evaluated to false value

    # todo: return some kind of weight or probability with matches
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

                # todo: it is temporary for keeping consistency with saved in trie old entity type schema
                tmap = ENT_MAPPING
                typed = groupby(lambda e: (tmap.get(e.ent_type) or e.ent_type) == span.label_, candidates)
                # typed = groupby(lambda entry: entry.ent_type == span.label_, candidates)

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

        # Some more checks on the best matches if there're several matches
        if len(best_matches) > 1:
            # best_matches = [max(best_matches, key=lambda entry: metric(raw_d(raw(entry.uri)), text))]
            best_matches = groupby(lambda entry: metric(raw_d(raw(entry.uri)), text), best_matches)
            best_matches = best_matches[max(best_matches)]
        return best_matches

    def _is_acronym(self, text, len_threshold=5):
        return len(text) < len_threshold and text.isupper()

    def _longest_prefix(self, text):
        l = len(text)
        left = max(1, floor(l * self._metric_threshold))
        for end in range(l, left-1, -1):
            if self._trie.has_subtrie(text[:end]):
                return text[:end]

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


def build_linker(linker, model_dir):
    from experiments.ontology.sub_ont import gf, gfall, dbr
    from experiments.data_utils import unpickle
    dataset_dir = '/home/user/datasets/dbpedia/ner/'
    dataset_file = 'crecords.v2.pck'
    crecords = list(unpickle(dataset_dir + dataset_file))
    data1 = set(from_crecords(crecords))

    graph = gfall
    data2 = list(from_graph(graph))

    for data in [data1, data2]:
        log.info('linker: updating data ({} records)'.format(len(data)))
        linker.update(data)
        linker.save(model_dir)
    test_linker_trie(linker)


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
    nlp = load_nlp()

    titles = [dbr.JetBrains, dbr.Microsoft_Windows]
    for i, title in enumerate(titles):
        print('\n')
        print(i, str(title))
        article = get_article(title)
        doc = nlp(article['text'])

        # Change (uncomment) the needed test method
        # test_get_candidates(linker, doc)
        # test_link(linker, doc)


def test_link(linker, doc):
    answers = linker.link(doc.ents, depth=2)
    for j, (sent, sent_ents) in enumerate(sentences_ents(doc)):
        print()
        print(j, sent)
        for ent in sent_ents:
            uri = answers.get(ent, '-')
            uri_str = raw(str(uri))
            print('{}: <{}>'.format(ent.text.rjust(20), uri_str))


def test_get_candidates(linker, doc):
    prev_sent = ''
    for ent in doc.ents:
        sent_text = ent.sent.text.strip()
        if sent_text != prev_sent:
            print()
            print(sent_text)
            prev_sent = sent_text

        entries = linker.get_candidates(ent)
        uris_str = '; '.join([raw(entry.uri) for entry in entries if entry])
        print('({}:{}) ({}) <{}> is [{}]'.format(ent.start, ent.end, ent.label_, ent.text, uris_str))


if __name__ == "__main__":
    from experiments.ontology.data_structs import ContextRecord, EntityRecord  # for unpickle()
    from experiments.ontology.symbols import ENT_CLASSES
    from experiments.nlp_utils import sentences_ents
    from experiments.ontology.config import config

    log.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=log.INFO)

    model_dir = config['models']['dir']
    assert os.path.isdir(model_dir)

    ntr = NERTypeResolver()
    linker = NERLinker(ner_type_resolver=ntr, metric_threshold=0.8, strict_type_match=False)

    # build_linker(linker, model_dir)

    linker.load(model_dir)
    test_linker(linker)

