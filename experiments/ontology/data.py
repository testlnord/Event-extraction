import logging as log
import os
import pickle
import random
import re
from collections import defaultdict
from copy import copy
from itertools import permutations, islice
from multiprocessing import Pool

from SPARQLWrapper.SPARQLExceptions import QueryBadFormed
from fuzzywuzzy import fuzz
from intervaltree import IntervalTree

from experiments.data_utils import unpickle
from experiments.nlp_utils import merge_ents_offsets, sentences_ents
from experiments.ontology.data_structs import RelationRecord, EntityRecord, ContextRecord, EntityMention
from experiments.ontology.linker import NERTypeResolver
from experiments.ontology.sub_ont import dbo, dbr, gf, gdb
from experiments.ontology.sub_ont import get_article, get_label


def filter_context(crecord):
    """Only chooses the sentence where the entities (subject and object) are present.
    Does not yield other sentences. Returns original crecord if it is valid."""
    ctext = crecord.context
    rex = '\n+'
    matches = [m.span() for m in re.finditer(rex, ctext)]
    ends, starts = zip(*matches) if len(matches) != 0 else ([], [])
    starts = [0] + list(starts)
    ends = list(ends) + [len(ctext)]
    spans = [(a, b) for a, b in zip(starts, ends) if a < b]

    itree = IntervalTree.from_tuples(spans)
    ssent = itree[crecord.s_startr:crecord.s_endr]
    if ssent == itree[crecord.o_startr:crecord.o_endr]:
        p = ssent.pop()
        cr = copy(crecord)
        cr.cut_context(p.begin, p.end)
        return cr


def filter_contexts(crecords):
    return list(filter(None, map(filter_context, crecords)))


def load_rc_data(allowed_classes, rc_file, rc_neg_file, neg_ratio=0., shuffle=True, exclude_records=frozenset()):
    """
    Load data for relation classification given paths to pickled RelationRecords and make basic filtering.
    :param allowed_classes: keep only these relation classes (specified in URI form)
    :param rc_file: path to file with pickled RelationRecords
    :param rc_neg_file: path to file with pickled negative RelationRecords (i.e. no relation)
    :param neg_ratio: max ratio of #negatives to #positive records (i.e. how much negatives to load)
    :param shuffle: bool, shuffle or not dataset
    :param exclude_records: collection (preferably set-like with fast lookup) of records to filter them out
    :return: List[RelationRecord]
    """
    _classes = set(allowed_classes)
    records = []
    for rr in unpickle(rc_file):
        # Same subject and object sometimes happen
        if (str(rr.relation) in _classes and rr not in exclude_records
            and rr.valid_offsets and rr.subject != rr.object):
            _fc = filter_context(rr)
            if _fc is not None:
                records.append(_fc)
    nb_neg = int(len(records) * neg_ratio)
    rc_neg_gen = (filter_context(rr) for rr in unpickle(rc_neg_file) if rr.valid_offsets and rr.subject != rr.object)
    rc_neg = islice(filter(None, rc_neg_gen), nb_neg)
    records.extend(rc_neg)
    if shuffle: random.shuffle(records)
    return records


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
            for start_char, end_char, surface_form in offsets_list:
                yield EntityMention(start_char, end_char, uri)


def resolve_relations(art_id, doc, ents_all, graph=gdb):
    """

    :param art_id: id of the article
    :param doc: spacy.Doc
    :param ents_all: List[EntityMention]
    :param graph: RDFLib.Graph to resolve relations
    :yield: RelationRecord
    """
    for sent, sent_ents in sentences_ents(doc, ents=ents_all):
        # Resolve relations: look at all possible pairs and try to resolve them
        # NB: multiple entries of the same entities could produce almost identical RelationRecords,
        #   i.e. same contexts and triples, only spans are different (but it is expected behaviour)
        for s, o in permutations(sent_ents, 2):
            r_uris = list(graph.predicates(s.uri, o.uri))
            if len(r_uris) == 0: r_uris.append('')
            for r_uri in r_uris:
                yield RelationRecord(s.uri, r_uri, o.uri, s.start_char, s.end_char, o.start_char, o.end_char,
                                     sent.text, sent.start_char, sent.end_char, source_id=art_id)


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


def run_extraction(nlp, inner_graph, outer_graph, subject_uris, use_all_articles=False, n_threads=7, batch_size=1000):
    """
    Implements specific policy of extraction (i.e. choice of candidate entities and articles)
    :param inner_graph: for example -- domain-specific graph
    :param outer_graph: the whole known 'world' to resolve entities from
    :param subject_uris: candidate entities to search for links with them
    :param use_all_articles:
    :param n_threads:
    :param batch_size:
    :yield: Tuple(article id, spacy.Doc, List[EntityMention])
    """
    visited = set()
    ls = len(subject_uris)
    for i, subject in enumerate(subject_uris, 1):
        log.info('subject #{}/{}: {}'.format(i, ls, subject))
        subj_article = get_article(subject)
        if subj_article is not None:
            # Search linked objects only in the provided, likely domain-specific 'world' (inner_graph)
            objects = set(inner_graph.objects(subject=subject))
            articles = map(get_article, objects) if use_all_articles else [subj_article]
            articles = [a for a in articles if a is not None and a['id'] not in visited]
            log.info('articles: {}; objects: {}'.format(len(articles), tuple(map(str, objects))))

            objects.add(subject)
            texts = [article['text'] for article in articles]
            docs = nlp.pipe(texts, n_threads=n_threads, batch_size=batch_size)
            for article, doc in zip(articles, docs):
                # Resolving entities from the whole 'world' (outer_graph)
                ents_article = list(resolve_entities(article['links'], graph=outer_graph))
                candidates = set(ent.uri for ent in ents_article).union(objects)
                ents_found = next(get_contexts([doc], *candidates))

                ents_all = merge_ents_offsets(ents_article, ents_found)
                if len(ents_all) > 0:
                    yield article['id'], doc, ents_all
                visited.add(article['id'])


# Writes all found relations. Filtering of only needed classes should be done later, in encoder, for example.
def make_dataset(nlp, ner_outfile, rc_outfile, rc_other_outfile, rc_no_outfile, inner_graph=gf, outer_graph=gdb):
    fner = open(ner_outfile, 'wb')
    frc = open(rc_outfile, 'wb')
    frc2 = open(rc_other_outfile, 'wb')
    frc0 = open(rc_no_outfile, 'wb')

    counter = defaultdict(int)
    try:
        subject_uris = set(inner_graph.subjects())
        total_rels = 0
        total_rels_filtered = 0
        total_ents = 0
        # NB: using outer_graph for the search of objects for subject_uris!
        # graph = outer_graph
        graph = inner_graph
        for art_id, doc, ents in run_extraction(nlp, graph, outer_graph, subject_uris, use_all_articles=False):
            cr = ContextRecord(doc.text, 0, len(doc.text), art_id, ents=None)
            ers = [EntityRecord(cr, *ent) for ent in ents]
            cr.ents = ers
            pickle.dump(cr, fner)

            nb_filtered = 0
            for j, rrel in enumerate(resolve_relations(art_id, doc, ents, graph=outer_graph), 1):
                counter[rrel.relation] += 1
                if not rrel.relation:
                    pickle.dump(rrel, frc0)
                # todo: make generic filtering function?
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


def transform_ner_dataset(nlp, crecords, allowed_ent_types, ner_type_resolver=NERTypeResolver(),
                          min_ents=0, min_ents_ratio=0., n_threads=8, batch_size=1000):
    """
    Transform dataset from ContextRecord-s format to spacy-friendly format (json), merging spacy entity types with ours.
    Data is yielded if at least one of the conditions (min_ents or min_ents_ration) is satisfied.
    By default all data is yielded (even if there're no entities).
    :param nlp: spacy.lang.Language
    :param crecords: dataset (iterable of ContextRecord-s)
    :param allowed_ent_types: what types to leave from spacy entity recogniser. Don't use spacy ner types altogether if empty
    :param ner_type_resolver: class governing mapping from entity uris to final entity types
    :param min_ents: minimum number of type-resolved entities in crecord to yield it
    :param min_ents_ratio: minimum ratio of number of type-resolved entities to number of tokens in crecord to yield it
    :param n_threads: n_threads parameter for nlp.pipe() and multiprocessing.Pool
    :param batch_size: batch_size parameter for nlp.pipe()
    :yield: list of json entities for spacy NER training (with already made Docs)
    """
    yielded = 0
    etypes = set(allowed_ent_types)
    docs = nlp.pipe([cr.context for cr in crecords], n_threads=n_threads, batch_size=batch_size)

    with Pool(processes=n_threads) as pool:
        for cr, doc in zip(crecords, docs):
            # Resolve entity types in parallel
            uris = [er.uri for er in cr.ents]
            ent_types = pool.map(ner_type_resolver.get_by_uri, uris)
            # ent_types = map(ner_type_resolver.get_by_uri, uris)  # sequential version

            ents = [(er.start, er.end, ent_type) for er, ent_type in zip(cr.ents, ent_types) if ent_type is not None]

            # Proceed only if there're enough entities
            nb_our_ents = len(ents)
            ents_ratio = nb_our_ents / len(doc)
            log_str = 'transform ner: ents_ratio: {:.2f}; our ents: {:4d}'.format(ents_ratio, nb_our_ents)
            if nb_our_ents >= min_ents or ents_ratio >= min_ents_ratio:
                # Add entities recognised by spacy if they aren't overlapping with any of our entities
                spacy_ents = [(e.start_char, e.end_char, e.label_) for e in doc.ents if e.label_ in etypes]
                ents = merge_ents_offsets(ents, spacy_ents)
                yield doc, ents

                yielded += 1
                log_str += ' (yielding #{:4d}: spacy ents: {:4d}; total ents: {:4d})'.format(yielded, len(spacy_ents), len(ents))
            log.info(log_str)


def test_resolve_relations(nlp, subject, relation, graph, test_all=False):
    objects = set(graph.objects(subject, relation))
    articles = [get_article(subject)]
    if test_all: articles.extend(map(get_article, objects))
    docs = list(nlp.pipe([art['text'] for art in articles]))

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


def repickle_rrecords(path):
    records = list(unpickle(path))
    with open(path, 'wb') as f:
        for rr in records:
            new_record = RelationRecord(rr.subject, rr.relation, rr.object, rr.s0, rr.s1, rr.o0, rr.o1, rr.context, rr.cstart, rr.cend, rr.source_id)
            pickle.dump(new_record, f)


if __name__ == "__main__":
    log.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=log.INFO)
    from experiments.ontology.config import config
    import spacy
    nlp = spacy.load(**config['models']['nlp'])

    # jbtriples = list(gf.triples((dbr.JetBrains, dbo.product, None)))
    # mtriples = list(gf.triples((dbr.Microsoft, dbo.product, None)))
    # test_resolve_relations(nlp, subject=dbr.Microsoft, relation=None, graph=gfall)

    data_dir = config['data']['dir']
    ner_out = os.path.join(data_dir, 'ner', 'crecords.v2.pck')
    rc_out = os.path.join(data_dir, 'rc', 'rrecords.v2.filtered.pck')
    rc0_out = os.path.join(data_dir, 'rc', 'rrecords.v2.negative.pck')
    rc2_out = os.path.join(data_dir, 'rc', 'rrecords.v2.other.pck')
    # make_dataset(nlp, ner_out, rc_out, rc2_out, rc0_out, inner_graph=gfall, outer_graph=gdb)

    # rc_paths = [rc_out, rc0_out, rc2_out]
    # for rc_path in rc_paths:
    #     repickle_rrecords(rc_path)


