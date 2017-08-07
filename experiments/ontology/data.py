import logging as log
import os
import csv
import pickle
from collections import defaultdict, namedtuple
from cytoolz import groupby
from itertools import permutations

import numpy as np
from fuzzywuzzy import fuzz
from intervaltree import IntervalTree
from rdflib import URIRef
from SPARQLWrapper.SPARQLExceptions import QueryBadFormed

from experiments.ontology.sub_ont import dbo, dbr, gdb
from experiments.ontology.sub_ont import get_article, get_label
from experiments.ontology.sub_ont import get_type, final_classes, get_final_class, get_superclasses_map
from experiments.ontology.ont_encoder import filter_context

# todo: move from globals
import spacy
nlp = spacy.load('en_core_web_sm')  # 'sm' for small
# from experiments.utils import load_nlp
# nlp = load_nlp()
# nlp = load_nlp(batch_size=32)


props_dir = '/home/user/datasets/dbpedia/qs/props/'
contexts_dir = '/home/user/datasets/dbpedia/contexts/'
classes_dir= '/home/user/datasets/dbpedia/qs/classes/'
superclasses_file = classes_dir + 'classes.map.json'


class RelationRecord:
    def __init__(self, s, r, o, s0, s1, o0, o1, ctext, cstart, cend, artid):
        # self.s = self.subject = URIRef(s)
        # self.r = self.relation = URIRef(r)
        # self.o = self.object = URIRef(o)
        self.s = self.subject = s
        self.r = self.relation = r
        self.o = self.object = o
        self.s0 = self.s_start = s0
        self.s1 = self.s_end = s1
        self.o0 = self.o_start = o0
        self.o1 = self.o_end = o1
        self.context = ctext
        self.cstart = cstart
        self.cend = cend
        self.article_id = artid

    def cut_context(self, begin, end):
        self.context = self.context[begin:end]
        self.cstart = self.cstart + begin
        self.cend = self.cstart + end
        self.s0 = self.s_start = max(self.cstart, self.s0)
        self.s1 = self.s_end = min(self.s1, self.cend)
        self.o0 = self.o_start = max(self.cstart, self.o0)
        self.o1 = self.o_end = min(self.o1, self.cend)

    @property
    def triple(self): return (self.subject, self.relation, self.object)

    @property
    def s_startr(self): return self.s_start - self.cstart  # offsets in dataset are relative to the whole document, not to the sentence

    @property
    def s_endr(self): return self.s_end - self.cstart

    @property
    def o_startr(self): return self.o_start - self.cstart

    @property
    def o_endr(self): return self.o_end - self.cstart

    @property
    def s_spanr(self): return (self.s_startr, self.s_endr)

    @property
    def o_spanr(self): return (self.o_startr, self.o_endr)


EntityMention = namedtuple('EntityMention', ['start', 'end', 'uri'])


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
        self.start = self.start + begin
        self.end = self.start + end
        for e in self.ents:
            e.cut_context(begin, end)

    @property
    def span(self):
        return self.start, self.end

    def json(self):
        return (self.article_id, self.span, self.context, [e.json for e in self.ents])

    def __str__(self):
        return self.context.strip() + '(' + '; '.join(str(e) for e in self.ents) + ')'



# for old data
def read_dataset(path):
    with open(path, 'r', newline='') as f:
        reader = csv.reader(f, delimiter=' ', quotechar='|', quoting=csv.QUOTE_NONNUMERIC)
        header = next(reader)
        log.info('read_dataset: header: {}'.format(header))
        # for s, r, o, s0, s1, o0, o1, ctext, cstart, cend, artid in reader:
        for data in reader:
            yield RelationRecord(*data)


def load_superclass_mapping(filename=classes_dir + 'classes.names.all.csv'):
    with open(filename, 'r', newline='') as f:
        reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
        names = [URIRef(name) for name, in reader]
        return get_superclasses_map(names)


def load_prop_superclass_mapping(filename=classes_dir + 'prop_classes.csv'):
    """Mapping between relations and superclasses"""
    classes = {}
    with open(filename, 'r', newline='') as f:
        reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC, delimiter=' ')
        for row in reader:
            icls, cls, rel = row[:3]
            if int(icls) >= 0:
                classes[rel] = cls
    return classes


def load_inverse_mapping(filename=classes_dir + 'prop_inverse.csv'):
    """Mapping between superclasses and their inverse superclasses"""
    inverse = {}
    with open(filename, 'r', newline='') as f:
        reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC, delimiter=' ')
        for icls, cls, iinv, inv in reader:
            inverse[cls] = inv
            inverse[inv] = cls
    return inverse


def load_all_data(classes, data_dir=contexts_dir, shuffle=True):
    """Load ContextRecords with classes from @classes from all files in the directory @data_dir and shuffle it."""
    dataset = []
    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        if os.path.isfile(filepath):
            for crecord in read_dataset(filepath):
                if crecord.relation in classes:
                    filtered = filter_context(crecord)
                    if filtered is not None:
                        dataset.append(filtered)
    if shuffle: np.random.shuffle(dataset)
    return dataset


def resolve_entities(article_links, graph=gdb):
    for resource_name, offsets_list in article_links.items():
        # Use only internal wiki links
        if not resource_name.startswith('http'):
            uri = dbr[resource_name]
            # Resolve uris of these filtered links
            try:
                next(graph.triples((uri, None, None)))
            except (StopIteration, QueryBadFormed):
                continue
            except Exception as e:  # sparqlstore cat throw excetion of generic type
                log.warning('resolve_entities: ignoring exception: {}'.format(e))
                continue
            for start, end, surface_form in offsets_list:
                yield EntityMention(start, end, uri)


def merge_ents_offsets(primal_ents, other_ents):
    """
    Merge ent lists with non-overlapping entries, giving precedence to primal_ents.
    :param primal_ents: iterable of tuples of form (begin_offset, end_offset, data)
    :param other_ents: iterable of tuples of form (begin_offset, end_offset, data)
    :return: merged list of ents
    """
    ents_tree = IntervalTree.from_tuples(e for e in primal_ents if e.end > e.start)
    ents_filtered = [ent for ent in other_ents if not ents_tree.overlaps(ent[0], ent[1])]
    ents_filtered.extend(primal_ents)
    return ents_filtered


# todo: need to filter repeated relations?
def resolve_relations(art_id, doc, ents_all, graph):
    """

    :param art_id: id of the article
    :param doc: spacy.Doc
    :param ents_all: List[EntityMention]
    :param graph: RDFLib.Graph to resolve relations
    :yield: RelationRecord
    """
    # Group entities by sentences
    sents_bound_tree = IntervalTree.from_tuples([(s.start_char, s.end_char, i) for i, s in enumerate(doc.sents)])
    def index(ent):  # help function for convenience
        # print(list(sorted([tuple(i) for i in sents_bound_tree.all_intervals], key=lambda t: t[0])))
        # print(ent)
        sents = sents_bound_tree[ent[0]:ent[1]]
        if sents: return sents.pop().data

    ents_in_sents = groupby(index, ents_all)
    for i, sent in enumerate(doc.sents):
        sent_ents = ents_in_sents.get(i, list())
        # Resolve relations: look at all possible pairs and try to resolve them
        for s, o in permutations(sent_ents, 2):
            r_uris = list(graph.predicates(s.uri, o.uri))
            if len(r_uris) == 0: r_uris.append('')
            for r_uri in r_uris:
                yield RelationRecord(s.uri, r_uri, o.uri, s.start, s.end, o.start, o.end,
                                     sent.text, sent.start_char, sent.end_char, artid=art_id)


def fuzzfind(doc, *candidates, metric=fuzz.ratio, metric_threshold=82):
    """
    Find all occurrences of candidates in doc and return their char offsets.
    Complexity (estimate): O(O(metric)*len(doc)*len(candidates))
    :param doc: spacy.Doc
    :param candidates: iterable of strings
    :param metric: mapping of form (str, str)->int
    :param metric_threshold: metric threshold
    :return: dict of form {candidate: list of char offsets} for found entities in doc
    """
    # Merge noun_chunks and entities to iterate over them along with the simple tokens
    for e in doc.noun_chunks: e.merge()
    for e in doc.ents: e.merge()
    found = defaultdict(list)
    for k, t in enumerate(doc):
        # Choosing 'better matching' entity
        raw_ent, measured = max([(c, metric(t, c)) for c in candidates], key=lambda tpl: tpl[1])
        if measured >= metric_threshold:
            ii = t.idx
            found[raw_ent].append((ii, ii+len(t)))
    return found


def get_contexts(docs, *ents_uris):
    """
    Search for occurrences of ents in the docs.
    :param docs: iterable of spacy.Doc
    :param ents_uris: uris of the ents to search for
    :yield: List[EntityMention] for each doc in @docs
    """
    raw_ents = {get_label(ent_uri): ent_uri for ent_uri in ents_uris}  # mapping from labels to original uris
    for doc in docs:
        ents_dict = fuzzfind(doc, *raw_ents.keys())
        ents_found = [EntityMention(a, b, raw_ents[raw_ent]) for raw_ent, offsets in ents_dict.items() for a, b in offsets]
        yield ents_found


def run_extraction(graph, subject_uris, use_all_articles=False, n_threads=7, batch_size=1000):
    """Implements specific policy of extraction (i.e. choice of candidate entities and articles)"""
    visited = set()
    ls = len(subject_uris)
    for i, subject in enumerate(subject_uris, 1):
        log.info('subject #{}/{}: {}'.format(i, ls, subject))
        subj_article = get_article(subject)
        if subj_article is not None:
            objects = set(graph.objects(subject=subject))
            articles = map(get_article, objects) if use_all_articles else [subj_article]
            articles = [a for a in articles if a is not None and a['id'] not in visited]
            log.info('articles: {}; objects: {}'.format(len(articles), tuple(map(str, objects))))

            objects.add(subject)
            texts = [article['text'] for article in articles]
            docs = nlp.pipe(texts, n_threads=n_threads, batch_size=batch_size)
            for article, doc in zip(articles, docs):
                ents_article = list(resolve_entities(article['links']))  # leaving default value of 'graph' parameter
                candidates = set(ent.uri for ent in ents_article).union(objects)
                ents_found = next(get_contexts([doc], *candidates))

                ents_all = merge_ents_offsets(ents_article, ents_found)
                if len(ents_all) > 0:
                    yield article['id'], doc, ents_all
                visited.add(article['id'])


# Writes all found relations. Filtering of only needed classes should be done later, in encoder, for example.
def make_dataset(ner_outfile, rc_outfile, rc_other_outfile, rc_no_outfile, graph):
    fner = open(ner_outfile, 'wb')
    frc = open(rc_outfile, 'wb')
    frc2 = open(rc_other_outfile, 'wb')
    frc0 = open(rc_no_outfile, 'wb')

    counter = defaultdict(int)
    try:
        subject_uris = set(graph.subjects())
        total_rels = 0
        total_rels_filtered = 0
        total_ents = 0
        for art_id, doc, ents in run_extraction(graph, subject_uris, use_all_articles=False):
            cr = ContextRecord(doc.text, 0, len(doc.text), art_id, ents=None)
            ers = [EntityRecord(cr, *ent) for ent in ents]
            cr.ers = ers

            pickle.dump(cr, fner)
            nb_filtered = 0
            for j, rrel in enumerate(resolve_relations(art_id, doc, ents, graph), 1):
                counter[rrel.relation] += 1
                if not rrel.relation:
                    pickle.dump(rrel, frc0)
                elif rrel.relation.startswith(dbo):
                    nb_filtered += 1
                    pickle.dump(rrel, frc)
                else:
                    pickle.dump(rrel, frc2)
            total_rels += j
            total_rels_filtered += nb_filtered
            total_ents += len(ents)
            log.info('article_id: {}; #ents: {} (total: {}); rels: {} (filtered: {}; total_filtered: {}; total: {})'
                     .format(art_id, len(ents), total_ents, j, nb_filtered, total_rels_filtered, total_rels))
    finally:
        print(dict(counter))
        fner.close()
        frc.close()
        frc2.close()
        frc0.close()


def transform_ner_dataset(nlp, crecords, allowed_ent_types, superclasses_map=dict(), n_threads=7, batch_size=1000):
    """
    Transform dataset from ContextRecord-s format to spacy-friendly format (json), merging spacy entity types with ours.
    :param nlp: spacy.lang.Language
    :param crecords: dataset (iterable of ContextRecord-s)
    :param allowed_ent_types: what types to leave from spacy entity recogniser. Don't use spacy ner types altogether if empty
    :param superclasses_map: buffer containing mapping from classes to final classes in the ontology hierarchy
    :param n_threads: n_threads parameter for nlp.pipe()
    :param batch_size: batch_size parameter for nlp.pipe()
    :return: list of json entities for spacy NER training (with already made Docs)
    """
    docs = nlp.pipe([cr.context for cr in crecords], n_threads=n_threads, batch_size=batch_size)
    etypes = set(allowed_ent_types)
    for cr, doc in zip(crecords, docs):
        ents = []
        for er in cr.ents:
            assert isinstance(er.uri, URIRef)
            _cls_uri = get_type(er.uri)
            cls_uri = superclasses_map.get(_cls_uri, get_final_class(_cls_uri))  # try to get the type from buffer
            if cls_uri is not None:
                superclasses_map[_cls_uri] = cls_uri  # add type to buffer (or do nothing useful if it is already there)
                ent_type = final_classes[cls_uri]
                ents.append((er.start, er.end, ent_type))
        # Add entities recognised by spacy if they aren't overlapping with any of our entities
        spacy_ents = [(e.start_char, e.end_char, e.label_) for e in doc.ents if e.label_ in etypes]
        log.info('transform ner: our ents: {}; merged spacy ents: {} (total spacy ents: {})'.format(len(ents), len(spacy_ents), len(doc.ents)))
        ents = merge_ents_offsets(ents, spacy_ents)
        if len(ents) > 0:
            yield doc, ents


def test_resolve_relations(nlp, subject, relation, graph, test_all=False):
    objects = set(graph.objects(subject, relation))
    articles = [get_article(subject)]
    if test_all: articles.extend(map(get_article, objects))
    docs = list(nlp.pipe([art['text'] for art in articles]))

    # for i, (art_id, doc, ents) in enumerate(zip(docs, get_contexts(docs, subject, *objects))):
    for i, (doc, ents_found) in enumerate(zip(docs, get_contexts(docs, subject, *objects))):
        article = articles[i]
        ents_article = list(resolve_entities(article['links']))
        ents_all = merge_ents_offsets(ents_found, ents_article)
        diff = set(ents_all) - set(ents_found)
        print('diff ents:', len(diff))
        print('doc #{}'.format(i))

        ents = ents_all  # to test all
        # ents = ents_found  # to test found
        # ents = ents_article  # to test only article's native ents

        total_local = 0
        for j, rrel in enumerate(resolve_relations(article['id'], doc, ents, graph)):
            if rrel.relation.startswith(dbo):  # only predicates from dbo namespace
                total_local += 1
                s = '{}({})'.format(j, total_local)
                print(s, repr(rrel.context.strip()))
                print(' '*len(s), *map(str, rrel.triple))
        print('total relations locally: {}'.format(total_local))


if __name__ == "__main__":
    log.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=log.INFO)

    from experiments.ontology.sub_ont import gf, gfall, gdb, gdbo

    # test_resolve_relations(nlp, subject=dbr.Microsoft, relation=None, graph=gfall)
    # exit()

    data_dir = '/home/user/datasets/dbpedia/'
    ner_out = os.path.join(data_dir, 'ner', 'crecords.v2.pck')
    rc_out = os.path.join(data_dir, 'rc', 'rrecords.v2.filtered.pck')
    rc2_out = os.path.join(data_dir, 'rc', 'rrecords.v2.other.pck')
    rc0_out = os.path.join(data_dir, 'rc', 'rrecords.v2.negative.pck')
    make_dataset(ner_out, rc_out, rc2_out, rc0_out, graph=gfall)

    # jbtriples = list(gf.triples((dbr.JetBrains, dbo.product, None)))
    # mtriples = list(gf.triples((dbr.Microsoft, dbo.product, None)))

    # Try reading the dataset
    # from experiments.data_utils import unpickle
    # for i, crecord in enumerate(unpickle(output_file)):
    #     count = all(isinstance(er.uri, URIRef) for er in crecord.ents)
    #     print(i, count)
