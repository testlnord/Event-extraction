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

from experiments.utils import except_safe


dbp_dir = '/home/user/datasets/dbpedia/'
basedir = '/home/user/datasets/dbpedia/z2016_10/'
# dir_articles = '/home/user/datasets/dbpedia/articles/'
dir_articles = '/home/user/datasets/dbpedia/articles3/'

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


# It can happen that Virtuoso server is at the process of making a checkpoint, which will result in the following exceptions.
# Checkpoint takes few seconds, so, the easy way is just to wait few seconds and try again. Decorator does exactly that.
@except_safe(EndPointNotFound, HTTPError)
def get_article(subject):
    """Fail-safe when article is not present."""
    try:
        id_uri = next(gdb.objects(subject=subject, predicate=dbo.wikiPageID))
    except StopIteration:
        return None
    try:
        with open(dir_articles + id_uri) as f:
            return json.load(f)
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
    direct_scls = list(gdbo.objects(uri, RDFS.subClassOf))
    if len(direct_scls) > 0:
        c = direct_scls[0]
        return uri if c == OWL.Thing else c
    return None


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


def query_raw(q):
    sparql = SPARQLWrapper(endpoint, update_endpoint)
    sparql.setHTTPAuth(DIGEST)
    sparql.setCredentials('dba', 'admin')
    sparql.setMethod(POST)
    sparql.setQuery(q)
    return sparql.query()


if __name__ == "__main__":
    log.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=log.INFO)

    print(NERTypeResolver.final_classes_names)
    exit()

    # Some simple tests
    gtest = ds.get_context('gtest')
    gtest.update('INSERT DATA {<s1> <r1> <o1> }')
    # gtest.add((URIRef('s1'), URIRef('r1'), URIRef('o1')))  # analogous
    print(len(gtest.query('SELECT * WHERE {?s ?r ?o}')))
    gtest.update('DELETE DATA {<s1> <r1> <o1> }')
    print(len(gtest.query('SELECT * WHERE {?s ?r ?o}')))



