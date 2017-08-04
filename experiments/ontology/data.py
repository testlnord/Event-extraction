import logging as log
import os
import csv
import pickle
import re
from collections import defaultdict

import numpy as np
from fuzzywuzzy import fuzz
from rdflib import URIRef

from experiments.ontology.sub_ont import raw, gfall, gf, dbo, dbr
from experiments.ontology.sub_ont import get_article, get_label
from experiments.ontology.sub_ont import get_type, final_classes, get_final_class, get_superclasses_map
from experiments.ontology.ont_encoder import filter_context

# todo: move from globals
import spacy
nlp = spacy.load('en_core_web_sm')  # 'sm' for small
# from experiments.utils import load_nlp
# nlp = load_nlp()
# nlp = load_nlp(batch_size=32)


# Read list of valid properties from the file
props_dir = '/home/user/datasets/dbpedia/qs/props/'
props = props_dir + 'all_props_nonlit.csv'
with open(props, 'r', newline='') as f:
    prop_reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
    valid_props = [URIRef(prop) for prop, n in prop_reader]


def validate(s, r, o):
    """Check if triple is one we want in the dataset."""
    return r in valid_props


contexts_dir = '/home/user/datasets/dbpedia/contexts/'
delimiter = ' '
quotechar = '|'
quoting = csv.QUOTE_NONNUMERIC


def make_dataset(triples, output_path, mode='w'):
    log_total = 0
    header = ['s', 'r', 'o', 's_start', 's_end', 'o_start', 'o_end', 'context', 'context_start', 'context_end', 'article_id']
    with open(output_path, mode, newline='') as f:
        writer = csv.writer(f, delimiter=delimiter, quotechar=quotechar, quoting=quoting)  # todo:adjust
        if mode == 'w': writer.writerow(header)
        for i, triple in enumerate(triples, 1):
            if validate(*triple):
                log.info('make_dataset: processing triple #{}: {}'.format(i, [str(t) for t in triple]))
                for j, (ctx, s0, s1, o0, o1, art) in enumerate(get_contexts(*triple), 1):
                    # write both the text and its' source
                    log_total += 1
                    log.info('make_dataset: contex #{} (total: {})'.format(j, log_total))
                    writer.writerow(list(triple) + [s0, s1, o0, o1, ctx.text.strip(), ctx.start_char, ctx.end_char] + [int(art['id'])])


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


from collections import namedtuple
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


def read_dataset(path):
    with open(path, 'r', newline='') as f:
        reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar, quoting=quoting)
        header = next(reader)
        log.info('read_dataset: header: {}'.format(header))
        # for s, r, o, s0, s1, o0, o1, ctext, cstart, cend, artid in reader:
        for data in reader:
            yield RelationRecord(*data)


### Auxiliary functions ###


classes_dir= '/home/user/datasets/dbpedia/qs/classes/'
superclasses_file = classes_dir + 'classes.map.json'


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


###  ###


def make_fuzz_metric(fuzz_ratio=80):
    def fz(t1, t2):
        return fuzz.ratio(t1, t2) >= fuzz_ratio
    return fz


# no rdf interaction
def fuzzfind_plain(doc, s, r, o, metric = make_fuzz_metric()):
    for e in doc.noun_chunks: e.merge()
    for e in doc.ents: e.merge()
    for k, sent in enumerate(doc.sents):
        s0 = s1 = o0 = o1 = -1
        for i, t in enumerate(sent):
            # todo: it is possible to match 'better matching' entity
            ii = t.idx
            if metric(t.text, s): s0, s1 = ii, ii+len(t)
            elif metric(t.text, o): o0, o1 = ii, ii+len(t)
            if s0 >= 0 and o0 >= 0:
                yield sent, s0, s1, o0, o1
                break


def get_contexts(s, r, o):
    stext = get_label(s)
    rtext = get_label(r)
    otext = get_label(o)
    s_article = get_article(s)
    if s_article is not None:
        sdoc = nlp(s_article['text'])
        for context in fuzzfind_plain(sdoc, stext, rtext, otext):
            yield (*context, s_article)
    # is_literal = (otext[0] == otext[-1] == '"')
    o_article = get_article(o)
    if o_article is not None:
        odoc = nlp(o_article['text'])
        for context in fuzzfind_plain(odoc, stext, rtext, otext):
            yield (*context, o_article)


# todo: support RelationRecord None relations
# todo: do run_extraction


def get_article_ents(article_links, graph):
    for resource_name, offsets_list in article_links.items():
        # Use only internal wiki links
        if not resource_name.startswith('http'):
            uri = dbr[resource_name]
            # Resolve uris of these filtered links
            #   could there be some problems with bad uri form?
            try:
                next(graph.triples((uri, None, None)))
            except StopIteration:
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
    ents_tree = IntervalTree.from_tuples(primal_ents)
    ents_filtered = [ent for ent in other_ents if not ents_tree.overlaps(ent[0], ent[1])]
    ents_filtered.extend(primal_ents)
    return ents_filtered


from cytoolz import groupby
from itertools import permutations
def resolve_relations(art_id, doc, ents_all, graph):
    """

    :param art_id: id of the article
    :param doc: spacy.Doc
    :param ents_all: List[EntityMention]
    :param graph: RDFLib.Graph to resolve relations
    :yield: RelationRecord
    """
    # Group entities by sentences
    sents_bound_tree = IntervalTree.from_tuples([(s.start_char_, s.end_char_, i) for i, s in enumerate(doc.sents)])
    def index(ent): return sents_bound_tree[ent[0]:ent[1]].pop().data  # help function for convenience
    ents_in_sents = groupby(index, ents_all)
    for i, sent in doc.sents:
        sent_ents = ents_in_sents[i]
        # Resolve relations: look at all possible pairs and try to resolve them
        for s, o in permutations(sent_ents, 2):
            r_uris = list(graph.predicates(s.uri, o.uri))
            if len(r_uris) == 0: r_uris = [None]
            for r_uri in r_uris:
                yield RelationRecord(s.uri, r_uri, o.uri, s.start, s.end, o.start, o.end,
                                     sent.text, sent.start_char, sent.end_char, artid=art_id)


def fuzzfind0(doc, *candidates, metric=fuzz.ratio, metric_threshold=82):
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


def get_contexts0(docs, *ents_uris):
    """
    Search for occurrences of ents in the docs.
    :param docs: iterable of spacy.Doc
    :param ents_uris: uris of the ents to search for
    :yield: (spacy.Doc, List[EntityMention])
    """
    raw_ents = {get_label(ent_uri): ent_uri for ent_uri in ents_uris}  # mapping from labels to original uris
    for doc in docs:
        ents_dict = fuzzfind0(doc, *raw_ents.keys())
        ents_found = [EntityMention(a, b, raw_ents[raw_ent]) for raw_ent, offsets in ents_dict.items() for a, b in offsets]
        yield doc, ents_found


def run_extraction(graph, subject_uris, visited=set(), use_all_articles=False, n_threads=7, batch_size=1000):
    """Implements specific policy of extraction (i.e. choice of candidate entities and articles)"""
    ls = len(subject_uris)
    for i, subject in enumerate(subject_uris, 1):
        log.info('subject #{}/{}: {}'.format(i, ls, subject))
        subj_article = get_article(subject)
        if subj_article is not None:
            objects = set(graph.objects(subject=subject))
            articles = filter(None, map(get_article, objects)) if use_all_articles else [subj_article]
            articles = list(filter(lambda a: a['id'] not in visited, articles))
            # todo: What articles to add to art_ids: all or only the main (subj_article)?
            visited.add(subj_article['id'])
            log.info('articles: {}; objects: {}'.format(len(articles), objects))

            texts = [article['text'] for article in articles]
            docs = nlp.pipe(texts, n_threads=n_threads, batch_size=batch_size)
            for article, (doc, ents_found) in zip(articles, get_contexts0(docs, subject, *objects)):
                ents_article_gen = get_article_ents(article['links'], graph)
                ents_all = merge_ents_offsets(ents_found, ents_article_gen)
                if len(ents_all) > 0:
                    yield article['id'], doc, ents_all


def ner_extractor(todo, graph):
    for art_id, doc, ents in run_extraction(todo, graph):
        cr = ContextRecord(doc.text, 0, len(doc.text), art_id, ents=None)
        ers = [EntityRecord(cr, *ent) for ent in ents]
        cr.ers = ers
        yield cr


def relation_extractor(todo, graph):
    for data in run_extraction(todo, graph):
        yield from resolve_relations(*data, graph)


# Writes all found relations. Filtering of only needed classes should be done later, in encoder, for example.
def pickle_writer(output_file, extractor):
    with open(output_file, 'wb') as f:
        for obj in extractor():
            pickle.dump(obj, f)


from intervaltree import IntervalTree, Interval
def transform_ner_dataset(nlp, crecords, allowed_ent_types, superclasses_map=dict(), n_threads=7, batch_size=500):
    """
    Transform dataset from ContextRecord-s format to spacy-friendly format (json), merging spacy entitiy types with ours.
    :param nlp: spacy.lang.Language
    :param crecords: dataset (iterable of ContextRecord-s)
    :param allowed_ent_types: what types to leave from spacy entity recogniser. Don't use spacy ner types altogether if empty
    :param superclasses_map: buffer containing mapping from classes to final classes in the ontology hierarchy
    :param n_threads: n_threads parameter for nlp.pipe()
    :param batch_size: batch_size parameter for nlp.pipe()
    :return: list of json entities for spacy NER training (with already made Docs)
    """
    sents = nlp.pipe([cr.context for cr in crecords], n_threads=n_threads, batch_size=batch_size)
    etypes = set(allowed_ent_types)
    for cr, sent in zip(crecords, sents):
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
        spacy_ents = [(e.start_char, e.end_char, e.label_) for e in sent.ents if e.label_ in etypes]
        log.info('transform ner: our ents: {}; merged spacy ents: {} (total spacy ents: {})'.format(len(ents), len(spacy_ents), len(sent.ents)))
        ents = merge_ents_offsets(ents, spacy_ents)
        if len(ents) > 0:
            yield sent, ents


### Tests ###


def test(triples):
    for triple in triples:
        ctxs = list(get_contexts(*triple))
        print(len(ctxs), triple, '\n')
        for i, ctx_data in enumerate(ctxs):
            print('_' * 40, i)
            print(ctx_data[0])


def test_count_data(valid_props=valid_props):
    # Count how much data did we get
    total = total_extracted = total_filtered = 0
    for prop in valid_props:
        print('\n\nPROPERTY:', prop)
        filename = contexts_dir+'test4_{}.csv'.format(raw(prop))
        data = list(read_dataset(filename))
        triples = list(gfall.triples((None, prop, None)))
        filtered = 0
        for i, crecord in enumerate(data, 1):
            # s, r, o, s0, s1, o0, o1, ctext, cstart, cend, artid = crecord
            if re.search('\n+', crecord.context.strip()):
                print(i, crecord.triple)
                print(crecord.context)
                fct = filter_context(crecord)
                filtered += bool(fct is None)
                print('<{}>'.format(fct.context if fct is not None else None))
        total_filtered += filtered
        # Count
        nb_all = len(triples)
        nb_extracted = len(data)
        print('{}: {}/{} ~{:.2f}'.format(prop, nb_extracted, nb_all, nb_extracted/nb_all))
        total += nb_all
        total_extracted += nb_extracted
    tt = total_extracted - total_filtered
    print('filtered/extracted: {}/{} ~{:.2f}'.format(total_filtered, total_extracted, total_filtered/total_extracted))
    print('total: {}/{} ~{:.2f}'.format(tt, total, tt/total))


def test_get_contexts(subject, relation):
    subj_article = get_article(subject)
    objects = set(gf.objects(subject, relation))
    print(objects)
    for i, crecord in enumerate(get_contexts0([subj_article], subject, *objects)):
        print()
        print(i, 'CONTEXT:', crecord.context.strip())
        for j, ent in enumerate(crecord.ents):
            print(j, 'ENT:', ent)


if __name__ == "__main__":
    log.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=log.INFO)

    # jbtriples = list(gf.triples((dbr.JetBrains, dbo.product, None)))
    # mtriples = list(gf.triples((dbr.Microsoft, dbo.product, None)))
    # test(jbtriples)

    # test_count_data()
    # test_get_contexts(dbr.Microsoft, dbo.product)

    # Make NER dataset
    visited = set()
    subject_uris = set(gf.subjects())
    output_file = '/home/user/datasets/dbpedia/ner/' + 'crecords.full.pck'
    try:
        visited = make_ner_dataset(output_file, subject_uris, visited, use_all_articles=True)
    except Exception as e:
        print(e)
    finally:
        print('visited articles (total: {}):'.format(len(visited), visited))

    # Try reading the dataset
    # from experiments.data_utils import unpickle
    # for i, crecord in enumerate(unpickle(output_file)):
    #     count = all(isinstance(er.uri, URIRef) for er in crecord.ents)
    #     print(i, count)

    # Make relation classification dataset
    # for prop in valid_props:
    #     triples = gfall.triples((None, prop, None))
    #     filename = contexts_dir+'test0_{}.csv'.format(raw(prop))
    #     make_dataset(triples, filename)
