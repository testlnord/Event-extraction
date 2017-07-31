import logging as log
from rdflib.graph import Dataset, Graph, ReadOnlyGraphAggregate
from rdflib.namespace import RDF, RDFS, OWL, FOAF, Namespace, URIRef
from rdflib.store import Store
from rdflib.plugins.stores.sparqlstore import SPARQLStore, SPARQLUpdateStore
from rdflib.plugins.stores.sparqlstore import SPARQLWrapper
from SPARQLWrapper import DIGEST, POST
from SPARQLWrapper.SPARQLExceptions import EndPointNotFound
from urllib.error import HTTPError

import json
from fuzzywuzzy import fuzz

from experiments.utils import except_safe


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

iri_dbo = 'http://dbpedia.org/ontology'
iri_dbpedia = 'http://dbpedia.org'
iri_labels = 'http://dbpedia.org/labels'
iri_field = 'field'
iri_more = 'field:more'
dbo = Namespace('http://dbpedia.org/ontology/')
dbr = Namespace('http://dbpedia.org/resource/')

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


@except_safe(EndPointNotFound, HTTPError)
def get_label(uri):
    t = list(glo.objects(uri, RDFS.label))
    t = raw(uri) if len(t) == 0 else str(t[0])
    return t.split('(')[0].strip(' _')  # remove disambiguations


@except_safe(EndPointNotFound, HTTPError)
def get_type(uri):  # todo: add handling of literal types?
    t = list(gdb.objects(uri, RDF.type))
    return None if len(t) == 0 else t[0]


@except_safe(EndPointNotFound, HTTPError)
def get_superclass(uri):
    """
    :param uri: uri of the class to find superclass of
    :return: uri of superclass, @uri if class has no superclass, else None (e.g. @uri is not a uri of the class)
    """
    direct_scls = gdbo.objects(uri, RDFS.subClassOf)
    if len(direct_scls) > 0:
        c = direct_scls[0]
        return uri if c == OWL.Thing else c
    return None


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


# todo: move from globals
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


### Classes ###


from experiments.ontology.symbols import ENT_CLASSES
final_classes = set(URIRef(dbo[s]) for s in ENT_CLASSES)


def get_final_class(cls):
    c = cls
    while not(c in final_classes or c is None):
        c = get_superclass(c)
        if c == cls:
            return None
    return c


# todo: need all possible @classes to map
# todo: save final mapping?
def get_superclasses_map(classes):
    superclasses = dict()
    for cls in classes:
        fc = get_final_class(cls)
        if fc is not None:
            superclasses[cls] = fc
    return superclasses


# todo: make dict-mapping of final uris to final ents
final_ents = {}

superclasses2ner_tags = {
    "Person": "PERSON",
    "Organisation": "ORG",
    "Place": "LOC",
    "Settlement": "GPE",
    "Country": "GPE",
    "Language": "LANGUAGE",
    "ProgrammingLanguage": None,
    "Work": "PRODUCT",
    "Software": None,
    "VideoGame": None,
}



def query_raw(q):
    sparql = SPARQLWrapper(endpoint, update_endpoint)
    sparql.setHTTPAuth(DIGEST)
    sparql.setCredentials('dba', 'admin')
    sparql.setMethod(POST)
    sparql.setQuery(q)
    return sparql.query()


if __name__ == "__main__":
    log.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=log.INFO)




