import logging as log
from rdflib.graph import Dataset, Graph, ReadOnlyGraphAggregate
from rdflib.namespace import RDF, RDFS, OWL, FOAF, Namespace, URIRef
from rdflib.store import Store
from rdflib.plugins.stores.sparqlstore import SPARQLStore, SPARQLUpdateStore
from rdflib.plugins.stores.sparqlstore import SPARQLWrapper
from SPARQLWrapper import DIGEST, URLENCODED, POSTDIRECTLY, POST, RDFXML, TURTLE
from SPARQLWrapper.SPARQLExceptions import EndPointNotFound
from urllib.error import HTTPError

import json
import csv
from collections import defaultdict
from fuzzywuzzy import fuzz

from experiments.utils import except_safe


log.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=log.INFO)
dbp_dir = '/home/user/datasets/dbpedia/'
basedir = '/home/user/datasets/dbpedia/z2016_10/'
# dir_articles = '/home/user/datasets/dbpedia/articles/'
dir_articles = '/home/user/datasets/dbpedia/articles2/'

update_endpoint = 'http://localhost:8890/sparql-auth'
# endpoint = 'http://localhost:8890/sparql'  # no update (insert, create graph, whatever...) access
endpoint = update_endpoint
store = SPARQLUpdateStore(endpoint, update_endpoint, autocommit=True)  # need to call store.commit explicitly todo: there's some trouble in sparqlstore's source with that autocommit logic
store.setHTTPAuth(DIGEST)
store.setCredentials(user='dba', passwd='admin')
ds = Dataset(store, default_union=False)

dbo = Namespace('http://dbpedia.org/ontology/')
dbr = Namespace('http://dbpedia.org/resource/')
iri_dbo = 'http://dbpedia.org/ontology'
iri_dbpedia = 'http://dbpedia.org'
iri_labels = 'http://dbpedia.org/labels'
iri_field = 'field'
iri_more = 'field:more'

# NB: for existing graphs use 'ds.get_context(iri)', for new graphs use 'ds.graph(iri)'
# errors or silent ignoring of actions (on non-existing graphs) otherwise should be expected...
# remove_graph also requires already existing graph
gdb = ds.get_context(iri_dbpedia)
gdbo = ds.get_context(iri_dbo)
glabels = ds.get_context(iri_labels)
glo = ReadOnlyGraphAggregate([gdbo, glabels])
gall = ReadOnlyGraphAggregate([gdb, gdbo])
gf = ds.get_context(iri_field)
gmore = ds.get_context(iri_more)
gfall = ReadOnlyGraphAggregate([gf, gmore])

# Some simple tests
# gtest = ds.get_context('gtest')
# gtest.update('INSERT DATA {<s1> <r1> <o1> }')
# gtest.add((URIRef('s1'), URIRef('r1'), URIRef('o1')))  # analogous
# print(len(gtest.query('SELECT * WHERE {?s ?r ?o}')))
# gtest.update('DELETE DATA {<s1> <r1> <o1> }')
# print(len(gtest.query('SELECT * WHERE {?s ?r ?o}')))


def add_triples(query):
    """
    Add triples from CONSTRUCT query to subgraph.
    Query can be of any complexity (using FROM and FROM NAMED simultaneously, for instance)
    """
    qres = ds.query(query)  # gen-r of results of the construct query
    for res in qres:
        gf.add(res)  # adding to graph 'gf' in RDF Database (context is the graph 'gf')

# It can happen that Virtuoso server is at the process of making a checkpoint, which will result in the following exceptions.
# Checkpoint takes few seconds, so, the easy way is just to wait few seconds and try again. Decorator does exactly that.
@except_safe(EndPointNotFound, HTTPError)
def get_article(subject):
    """Fail-safe, when article is not present."""
    try:
        id_uri = next(gdb.objects(subject=subject, predicate=dbo.wikiPageID))
    except StopIteration:
        return None
    try:
        with open(dir_articles + id_uri) as f:
            art = json.load(f)
        text = art['text']
        first_par = text.find('\n\n')  # cut the title
        art['text'] = text[first_par+2:]
        return art
    except FileNotFoundError:
        return None

def raw(uri):
    return uri.rsplit('/', 1)[-1]

def get_labels(): # not ready!
    res = glo.subject_objects(RDFS.label)
    return dict([(uri, str(t).split('(')[0].strip(' _')) for uri, t in res])

@except_safe(EndPointNotFound, HTTPError)
def get_label(uri):
    t = list(glo.objects(uri, RDFS.label))
    t = raw(uri) if len(t) == 0 else str(t[0])
    return t.split('(')[0].strip(' _')  # remove disambiguations

def make_fuzz_metric(fuzz_ratio=80):
    def fz(t1, t2):
        return fuzz.ratio(t1, t2) >= fuzz_ratio
    return fz

def make_sim_metric(similarity_threshold):
    def sm(t1, t2):
        return t1.similarity(t2) >= similarity_threshold
    return sm

def make_metric(ratio=80, partial_ratio=95):
    def m(x, y):
        fzr = fuzz.ratio(x, y)
        fzpr = fuzz.partial_ratio(x, y)
        return (fzr >= ratio) or\
               (fzpr >= partial_ratio and fzr >= 0.6)
    return m

metric = make_fuzz_metric()

# no rdf interaction
def fuzzfind_plain(doc, s, r, o):
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

from spacy.en import English
nlp = English()
# from experiments.utils import load_nlp
# nlp = load_nlp()
# nlp = load_nlp(batch_size=32)

def get_contexts(s, r, o):
    stext = get_label(s)
    rtext = get_label(r)
    otext = get_label(o)
    s_article = get_article(s)
    if s_article is not None:
        sdoc = nlp(s_article['text'])
        for context in fuzzfind_plain(sdoc, stext, rtext, otext):
            yield (*context, s_article)
    # todo: if object is a literal, then maybe we should search by (s, r) pair and literal's type
    # is_literal = (otext[0] == otext[-1] == '"')
    o_article = get_article(o)
    if o_article is not None:
        odoc = nlp(o_article['text'])
        for context in fuzzfind_plain(odoc, stext, rtext, otext):
            yield (*context, o_article)


### Dataset management. key (input) points: output_path and valid_props.

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
        for i, triple in enumerate(triples):
            if validate(*triple):
                log.info('make_dataset: processing triple #{}: {}'.format(i, [str(t) for t in triple]))
                for j, (ctx, s0, s1, o0, o1, art) in enumerate(get_contexts(*triple), 1):
                    # write both the text and its' source
                    log_total += 1
                    log.info('make_dataset: contex #{} (total: {})'.format(j, log_total))
                    writer.writerow(list(triple) + [s0, s1, o0, o1, ctx.text.strip(), ctx.start_char, ctx.end_char] + [int(art['id'])])

# todo:
def read_dataset(path):
    with open(path, 'r', newline='') as f:
        reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar, quoting=quoting)
        header = next(reader)
        # for s, r, o, s0, s1, o0, o1, ctext, cstart, cend, artid in reader:
        for data in reader:
            yield data

##### Test
def test(triples):
    for triple in triples:
        ctxs = list(get_contexts(*triple))
        print(len(ctxs), triple, '\n')
        for i, ctx_data in enumerate(ctxs):
            print('_' * 40, i)
            print(ctx_data[0])

def query_raw(q):
    sparql = SPARQLWrapper(endpoint, update_endpoint)
    sparql.setHTTPAuth(DIGEST)
    sparql.setCredentials('dba', 'admin')
    sparql.setMethod(POST)
    sparql.setQuery(q)
    return sparql.query()

if __name__ == "__main__":
    # jbtriples = list(gf.triples((dbr.JetBrains, dbo.product, None)))
    # mtriples = list(gf.triples((dbr.Microsoft, dbo.product, None)))
    # test(jbtriples)
    # exit()

    # For every valid prop make a distinct file
    # from multiprocessing import Pool
    # triples = [gfall.triples((None, prop, None)) for prop in valid_props]
    # filenames = [contexts_dir+'test4_{}.csv'.format(raw(prop)) for prop in valid_props]
    # args = list(zip(triples, filenames))
    # with Pool(processes=len(valid_props)) as pool:
    #     nones = [pool.apply_async(make_dataset, _args) for _args in args]
    #     ress = [none.get() for none in nones]

    # for prop in valid_props:
    #     triples = gfall.triples((None, prop, None))
    #     make_dataset(triples, contexts_dir+'test3_{}.csv'.format(raw(prop)))

    prop = dbo.product
    triples = gfall.triples((None, prop, None))
    make_dataset(triples, contexts_dir+'test3_{}.csv'.format(raw(prop)))
    exit()

    classes_file = props_dir + 'prop_classes.csv'
    classes = {}
    with open(classes_file, 'r', newline='') as f:
        reader = csv.reader(f, quotechar=csv.QUOTE_NONNUMERIC)
        for cls, rel, _ in reader:
            if int(cls) >= 0:
                classes[rel] = int(cls)

